/*
 * Copyright (c) 2020 Barcelona Supercomputing Center
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Cristóbal Ramírez
 */

#include "cpu/vector_engine/vector_engine.hh"

#include <cassert>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>

#include "base/logging.hh"
#include "base/types.hh"
#include "cpu/translation.hh"
#include "debug/VectorEngine.hh"
#include "debug/VectorEngineInfo.hh"
#include "debug/VectorInst.hh"
#include "sim/faults.hh"
#include "sim/sim_object.hh"

VectorEngine::VectorEngine(VectorEngineParams *p) :
SimObject(p),
vector_config(p->vector_config),
VectorCacheMasterId(p->system->getMasterId(this, name() + ".vector_cache")),
vectormem_port(name() + ".vector_mem_port", this, p->vector_rf_ports),
vector_reg(p->vector_reg),
uniqueReqId(0),
num_clusters(p->num_clusters),
num_lanes(p->num_lanes),
vector_rob(p->vector_rob),
vector_lane(p->vector_lane),
vector_memory_unit(p->vector_memory_unit),
vector_inst_queue(p->vector_inst_queue),
vector_rename(p->vector_rename),
vector_reg_validbit(p->vector_reg_validbit),
last_vtype(0),
last_vl(0)
{
    //create independent ports
    for (uint8_t i=0; i< p->vector_rf_ports; ++i) {
        VectorRegMasterIds.push_back(p->system->getMasterId(this, name()
            + ".vector_reg" + std::to_string(i)));
        VectorRegPorts.push_back(VectorRegPort(name()
            + ".vector_reg_port", this, i));
    }

    DPRINTF(VectorEngineInfo,"VPU created:\n");
    DPRINTF(VectorEngineInfo,"Vector Renaming Enabled\n");
    DPRINTF(VectorEngineInfo,"Number of Physical Registers: %d \n"
        ,vector_rename->PhysicalRegs );
    DPRINTF(VectorEngineInfo,"Maximum VL: %d-bits\n"
        , vector_config->get_max_vector_length_bits() );
    DPRINTF(VectorEngineInfo,"Register File size: %dKB\n"
        , (float)vector_reg->get_size()/1024.0 );
    DPRINTF(VectorEngineInfo,"Vector Reorder buffer size: %d\n"
        , vector_rob->ROB_Size );
    DPRINTF(VectorEngineInfo,"Vector Arithmetic Queue Size: %d \n"
        ,vector_inst_queue->vector_arith_queue_size );
    DPRINTF(VectorEngineInfo,"Vector Memory Queue Size: %d \n"
        ,vector_inst_queue->vector_mem_queue_size );
    if (vector_inst_queue->OoO_queues)
        DPRINTF(VectorEngineInfo,"Out-of-Order Issue Logic\n");
    else
        DPRINTF(VectorEngineInfo,"In-Order Issue Logic\n");
    if (num_lanes<num_clusters) {
        panic("Invalid VPU Configuration\n");
    }
    DPRINTF(VectorEngineInfo,"Number of Clusters: %d\n",num_clusters);
    DPRINTF(VectorEngineInfo,"Lanes per Cluster: %d\n",num_lanes/num_clusters);
    DPRINTF(VectorEngineInfo,"Vector Memory Unit Port connected to l2 bus\n");
}

VectorEngine::~VectorEngine()
{
}


void
VectorEngine::regStats()
{
    SimObject::regStats();

    VectorArithmeticIns
        .name(name() + ".VectorArithmeticIns")
        .desc("Number vector arithmetic instructions");
    VectorMemIns
        .name(name() + ".VectorMemIns")
        .desc("Number vector arithmetic instructions");
    VectorConfigIns
        .name(name() + ".VectorConfigIns")
        .desc("Number vector configuration instructions");
    SumVL
        .name(name() + ".SumVL")
        .desc("Sum of all instruction's vector length to obtain the average ");
    AverageVL
        .name(name() + ".AverageVL")
        .desc("SumVL divided by the number of instructions")
        .precision(1);
    AverageVL = SumVL/(VectorMemIns+VectorArithmeticIns);

}

bool
VectorEngine::requestGrant(RiscvISA::VectorStaticInst *insn)
{
    bool rob_entry_available = !vector_rob->rob_full();
    bool queue_slots_available = ((insn->isVectorInstArith()
        || insn->isVecConfig()) && !vector_inst_queue->arith_queue_full())
        || (insn->isVectorInstMem()
        && !vector_inst_queue->mem_queue_full());

    /* Usually, the Vector engine must ensure to have at least 1 physical register to accept an instruction.
     * However, for LMUL>1 configurations this is different. When a vector operation with LMUL=2 means that 
     * the vector engine must have at least 2 physical registers available in order to acceept the instruction.
     * When LMUL=4, means that the vector engine must have at least 4 physical registers available in order 
     * to accept the instruction. And finally when LMUL=8, means that the vector engine must have at least 
     * 8 physical registers available in order to accept the instruction.
     * In the case 
     *
     * The vector INT/FP compare instructions compare two source operands and write the comparison result to a
     * mask register. The destination mask vector is always held in a single vector register,
     * with a layout of elements as described in Section Mask Register Layout.
     *
     * Vector mask-register logical operations operate on mask registers. The size of one element in a mask register
     * is SEW/LMUL, so these instructions all operate on single vector registers regardless of the setting of the vl-
     * mul eld in vtype. They do not change the value of vlmul.
     */ 
    bool mask_dst = insn->isMaskDst();

    uint8_t lmul = vector_config->get_vtype_lmul(last_vtype);

    bool enough_physical_regs = ((lmul == 1) || mask_dst) ? vector_rename->frl_elements() >= 1 :
                                (lmul == 2) ? vector_rename->frl_elements() >= 2 :
                                (lmul == 4) ? vector_rename->frl_elements() >= 4 :
                                (lmul == 8) ? vector_rename->frl_elements() >= 8 : 0;
    return  enough_physical_regs && queue_slots_available
        && rob_entry_available;
    //return  !vector_rename->frl_empty() && queue_slots_available
    //    && rob_entry_available;
}

bool
VectorEngine::isOccupied()
{
    bool cluster_ocuppied = 0;
    for (int i=0 ; i< num_clusters ; i++)
    {
        cluster_ocuppied = cluster_ocuppied || vector_lane[i]->isOccupied();
    }

    bool data_in_queues = 0;
    data_in_queues = (vector_inst_queue->Instruction_Queue.size() > 0)
        || (vector_inst_queue->Memory_Queue.size() > 0);

    bool rob_empty = vector_rob->rob_empty();

    return (cluster_ocuppied || data_in_queues
        || vector_memory_unit->isOccupied()
        || vector_inst_queue->isOccupied() || !rob_empty);
}

bool
VectorEngine::cluster_available()
{
    bool available = 0;

    for (int i=0 ; i< num_clusters ; i++)
    {
        available = available || !vector_lane[i]->isOccupied();
    }

    return available;
}

void
VectorEngine::printConfigInst(RiscvISA::VectorStaticInst& insn, uint64_t src1,uint64_t src2)
{
    uint64_t pc = insn.getPC();
    DPRINTF(VectorInst,"%s vl:%d, vtype:%d       PC 0x%X\n",insn.getName(),src1,src2,*(uint64_t*)&pc );
}

void
VectorEngine::printMemInst(RiscvISA::VectorStaticInst& insn,VectorDynInst *vector_dyn_insn)
{
    uint64_t pc = insn.getPC();
    bool gather_op = (insn.mop() ==3);

    uint32_t PDst = vector_dyn_insn->get_PDst();
    uint32_t POldDst = vector_dyn_insn->get_POldDst();
    uint32_t Pvs2 = vector_dyn_insn->get_PSrc2();
    uint32_t PMask = vector_dyn_insn->get_PMask();

    std::stringstream mask_ren;
    if (masked_op) {
        mask_ren << "v" << PMask << ".m";
    } else {
        mask_ren << "";
    }

    if (insn.isLoad())
    {
        if (gather_op){
            DPRINTF(VectorInst,"%s v%d v%d       PC 0x%X\n",insn.getName(),insn.vd(),insn.vs2(),*(uint64_t*)&pc);
            DPRINTF(VectorRename,"renamed inst: %s v%d v%d %s  old_dst v%d ,  \n",insn.getName(),PDst,Pvs2,mask_ren.str(),POldDst);
        } else {
            DPRINTF(VectorInst,"%s v%d       PC 0x%X\n",insn.getName(),insn.vd(),*(uint64_t*)&pc);
            DPRINTF(VectorRename,"renamed inst: %s v%d %s  old_dst v%d\n",insn.getName(),PDst,mask_ren.str(),POldDst);
        }
    }
    else if (insn.isStore())
    {
        DPRINTF(VectorInst,"%s v%d       PC 0x%X\n",insn.getName(),insn.vd(),*(uint64_t*)&pc );
        DPRINTF(VectorRename,"renamed inst: %s v%d %s\n",insn.getName(),PDst,mask_ren.str());
    } else {
        panic("Invalid Vector Instruction insn=%#h\n", insn.machInst);
    }
}

void
VectorEngine::printArithInst(RiscvISA::VectorStaticInst& insn,VectorDynInst *vector_dyn_insn,uint64_t src1)
{
    uint64_t pc = insn.getPC();
    std::string masked;
    if(masked_op) {masked = "v0.m";} else {masked = "   ";}
    std::string reg_type;
    if(insn.VectorToScalar()==1) {reg_type = "x";} else {reg_type = "v";}

    std::string scr1_type;
    scr1_type = (vx_op) ? "x" :
                (vf_op) ? "f" :
                (vi_op) ? " " : "v";

    uint32_t PDst = (insn.VectorToScalar()==1) ? insn.vd() : vector_dyn_insn->get_PDst();
    uint32_t POldDst = vector_dyn_insn->get_POldDst();
    uint32_t Pvs1 = (vx_op || vf_op || vi_op) ? insn.vs1() : vector_dyn_insn->get_PSrc1();
    uint32_t Pvs2 = vector_dyn_insn->get_PSrc2();
    uint32_t PMask = vector_dyn_insn->get_PMask();

    std::stringstream mask_ren;
    if (masked_op) {
        mask_ren << "v" << PMask << ".m";
    } else {
        mask_ren << "";
    }

    if (insn.arith1Src()) {
        DPRINTF(VectorInst,"%s %s%d v%d %s           PC 0x%X\n",insn.getName(),reg_type,insn.vd(),insn.vs2(),masked,*(uint64_t*)&pc );
        DPRINTF(VectorRename,"renamed inst: %s %s%d v%d %s  old_dst v%d\n",insn.getName(),reg_type,PDst,Pvs2,mask_ren.str(),POldDst);
    }
    else if (insn.arith2Srcs()) {
        DPRINTF(VectorInst,"%s %s%d v%d %s%d %s       PC 0x%X\n",insn.getName(),reg_type,insn.vd(),insn.vs2(),scr1_type,insn.vs1(),masked,*(uint64_t*)&pc );
        DPRINTF(VectorRename,"renamed inst: %s %s%d v%d %s%d %s  old_dst v%d\n",insn.getName(),reg_type,PDst,Pvs2,scr1_type,Pvs1,mask_ren.str(),POldDst);
    }
    else if (insn.arith3Srcs()) {
        DPRINTF(VectorInst,"%s %s%d v%d %s%d %s       PC 0x%X\n",insn.getName(),reg_type,insn.vd(),insn.vs2(),scr1_type,insn.vs1(),masked,*(uint64_t*)&pc );
        DPRINTF(VectorRename,"renamed inst: %s %s%d v%d %s%d v%d %s old_dst v%d\n",insn.getName(),reg_type,PDst,Pvs2,scr1_type,Pvs1,POldDst,mask_ren.str(),POldDst);
    } else {
        panic("Invalid Vector Instruction insn=%#h\n", insn.machInst);
    }
}



void
VectorEngine::renameVectorInst(RiscvISA::VectorStaticInst& insn, VectorDynInst *vector_dyn_insn, uint8_t micro_op_number)
{
    //vector_rename->print_rat();
    /* A vector mask occupies only one vector register regardless of SEW and LMUL.
     * The mask bits that are used for each vector operation depends on the current
     * SEW and LMUL setting.
     */
    bool mask_dst = insn.isMaskDst();
    /* Vector comparison instruction  */
    uint8_t dst_increase = (mask_dst) ? 0 : micro_op_number;
    /* Logical registers */
    uint64_t vd;
    uint64_t vs1,vs2;
    vd = insn.vd() + dst_increase;
    vs1 = insn.vs1()  + micro_op_number;
    vs2 = insn.vs2()  + micro_op_number;

    masked_op = (insn.vm()==0);

    vx_op = (insn.func3()==4) || (insn.func3()==6);
    vf_op = (insn.func3()==5);
    vi_op = (insn.func3()==3);

    uint8_t mop = insn.mop();
    bool gather_op = (mop ==3);

    if (insn.isVectorInstMem()) {
            /* Physical registers */
            Pvs1 = 1024;
            Pvs2 = 1024;
            PMask = 1024;
            PDst = 1024;
            POldDst = 1024;
        if (insn.isLoad()) {
            if (gather_op) {
                Pvs2 = vector_rename->get_preg_rat(vs2);
                vector_dyn_insn->set_PSrc2(Pvs2);
            }
            /* Physical  Mask */
            PMask = masked_op ? vector_rename->get_preg_rat(0) :1024;
            vector_dyn_insn->set_PMask(PMask);
            /* Physical Destination */
            PDst = vector_rename->get_frl();
            vector_dyn_insn->set_PDst(PDst);
            /* Physical Old Destination */
            POldDst = vector_rename->get_preg_rat(vd);
            vector_dyn_insn->set_POldDst(POldDst);
            vector_dyn_insn->setPOldDstValid(1);
            /* Setting the New Destination in the RAT structure */
            vector_rename->set_preg_rat(vd,PDst);
            /* Setting to 0 the new physical destinatio valid bit*/
            vector_reg_validbit->set_preg_valid_bit(PDst,0);
            DPRINTF(VectorValidBit,"Set Valid-bit %d : %d\n",PDst,0);
        }
        else if (insn.isStore()) {
            // TODO: maked stores are not implemented
            /* Physical Destination corresponds to the source for store operations */
            PDst = vector_rename->get_preg_rat(vd);
            /* Physical Destination */
            vector_dyn_insn->set_PDst(PDst);
        }
    }
    else if (insn.isVectorInstArith()) {
        /* Physical  Mask */
        PMask = masked_op ? vector_rename->get_preg_rat(0) :1024;
        vector_dyn_insn->set_PMask(PMask);
        /* When the instruction creates a mask, only the first microoperation whould get
         * a new register from the frl. Next microoperations must use the same since
         * mask vector registers are always allocated in only one register no matter LMUL.
         */

        /* Physical Destination */
        PDst =  (mask_dst && (micro_op_number>0)) ? PDst:
                (dst_write_ena) ? vector_rename->get_frl() :1024;
        vector_dyn_insn->set_PDst(PDst);
        /* Physical Old Destination */
        POldDst = (mask_dst && (micro_op_number>0)) ? POldDst:
                  (dst_write_ena) ? vector_rename->get_preg_rat(vd) : 1024;
        vector_dyn_insn->set_POldDst(POldDst);
        bool old_dst_valid = (mask_dst && (micro_op_number<(lmul-1))) ? 0:
                             dst_write_ena ? 1:0;
        vector_dyn_insn->setPOldDstValid(old_dst_valid);
        /* Physical source 2 */
        Pvs2 = vector_rename->get_preg_rat(vs2);
        vector_dyn_insn->set_PSrc2(Pvs2);
        /* When the instruction use an scalar value as source 1, the physical source 1 is disable
         * When the instruction uses only 1 source (insn.arith1Src()), the  source 1 is disable
         */
         /* Physical source 1 */
        Pvs1 = (!(vx_op || vf_op || vi_op) && !insn.arith1Src()) ? vector_rename->get_preg_rat(vs1) : 1024;
        vector_dyn_insn->set_PSrc1(Pvs1);
        /* dst_write_ena is set when the instruction has a vector destination register */
        if(dst_write_ena) {
            /* Setting the New Destination in the RAT structure */
            vector_rename->set_preg_rat(vd,PDst);
            /* Setting to 0 the new physical destinatio valid bit*/
            vector_reg_validbit->set_preg_valid_bit(PDst,0);
            DPRINTF(VectorValidBit,"Set Valid-bit %d: %d\n",PDst,0);
        }
    } else {
            panic("Invalid Vector Instruction insn=%#h\n", insn.machInst);
    }
}

void
VectorEngine::dispatch(RiscvISA::VectorStaticInst& insn, ExecContextPtr& xc,
    uint64_t src1,uint64_t src2,std::function<void()> dependencie_callback)
{
    /* Vector configuration instructions configure the registers last_vtype
     * and last_vl. Next instructions takes the configuration from those
     * registers, and carry out the values to be executed.
     * By doing this, out-of-order issue scheme does not care about the vector
     * configuration instructions.
     */
    if (insn.isVecConfig()) {
        last_vtype = src2;
        last_vl = src1;
        dependencie_callback();
        printConfigInst(insn,src1,src2);
        VectorConfigIns++;
        return;
    }

    /* Be sure that the instruction was added to some group in base.isa */
    if (insn.isVectorInstArith()) {
        assert( insn.arith1Src() | insn.arith2Srcs() | insn.arith3Srcs() );
    }

    if(insn.isWidening() || insn.isNarrowing()){
        panic("Widening/Narrowing vector instructions are not fully suported \n");
    }

    if ((vector_inst_queue->Instruction_Queue.size()==0)
        && (vector_inst_queue->Memory_Queue.size()==0)) {
        vector_inst_queue->startTicking(*this/*,dependencie_callback*/);
    }
    /* vector arithmetic operation which write a vector register*/
    dst_write_ena = !insn.VectorToScalar();

    /* When LMUL is bigger than 1, the instruction is splited in several micro operations.
     * Each micro operation has its own destination register and source operands
     * for LMUL=8 it is needed to assign 8 physical registers
     * for LMUL=4 it is needed to assign 4 physical registers
     * for LMUL=2 it is needed to assign 2 physical registers
     * for LMUL=1 it is needed to assign 1 physical register
     */
    lmul =  insn.VectorToScalar()       ?   1:
            insn.VectorMaskLogical()    ?   1:
            vector_config->get_vtype_lmul(last_vtype);

    if(insn.is_slide() && (lmul>1)) {
        panic("Slide with lmul>1 is not suported \n");
    }

    micro_op_vl_accum = last_vl;

    for(uint8_t i=0; i<lmul; i++) {
        VectorDynInst *vector_dyn_insn = new VectorDynInst(&insn,i);
        renameVectorInst(insn,vector_dyn_insn,i);
        
        DPRINTF(VectorEngine,"Creating Vector Dynamic instruction %s  microoperation number %d\n" ,vector_dyn_insn->get_VectorStaticInst()->getName() , vector_dyn_insn->getMicroOpNumber() );

        if (vector_rob->rob_empty()) {
            vector_rob->startTicking(*this);
        }

        /* When LMUL>1, each microoperation has its own vl */
        micro_op_vl_accum = micro_op_vl_accum - vector_config->get_max_vector_length_elem(last_vtype);
        uint64_t vl_iter = ((i+1)* vector_config->get_max_vector_length_elem(last_vtype));
        uint64_t mvl = vector_config->get_max_vector_length_elem(last_vtype);
        micro_op_vl = (last_vl >= vl_iter ) ? mvl : micro_op_vl_accum;

        if (insn.isVectorInstMem()) {
            if(i==0) {dependencie_callback();}
            uint32_t rob_entry = vector_rob->set_rob_entry(
                vector_dyn_insn->get_POldDst(), insn.isLoad());
            vector_dyn_insn->set_rob_entry(rob_entry);
            vector_inst_queue->Memory_Queue.push_back(
                new InstQueue::QueueEntry(insn,vector_dyn_insn,xc,
                    NULL,src1,src2,last_vtype,micro_op_vl));
            printMemInst(insn,vector_dyn_insn);
        }
        else if (insn.isVectorInstArith()) {
            if (dst_write_ena) {
                if(i==0) {dependencie_callback();}
                uint32_t rob_entry = vector_rob->set_rob_entry(
                    vector_dyn_insn->get_POldDst() , vector_dyn_insn->getPOldDstValid());
                vector_dyn_insn->set_rob_entry(rob_entry);
                vector_inst_queue->Instruction_Queue.push_back(
                    new InstQueue::QueueEntry(insn,vector_dyn_insn,xc,
                    NULL,src1,src2,last_vtype,micro_op_vl));
            } else {
                uint32_t rob_entry = vector_rob->set_rob_entry(0 , 0);
                vector_dyn_insn->set_rob_entry(rob_entry);
                vector_inst_queue->Instruction_Queue.push_back(
                    new InstQueue::QueueEntry(insn,vector_dyn_insn,xc,
                    dependencie_callback,src1,src2,last_vtype,micro_op_vl));
            }
            printArithInst(insn,vector_dyn_insn,src1);
        } else {
            panic("Invalid Vector Instruction, insn=%X\n", insn.machInst);
        }
    }
}

void
VectorEngine::issue(RiscvISA::VectorStaticInst& insn,VectorDynInst *dyn_insn,
    ExecContextPtr& xc ,uint64_t src1 ,uint64_t src2,uint64_t vtype,
    uint64_t vl, std::function<void(Fault fault)> done_callback) {

    //uint64_t pc = insn.getPC();
    
    if (insn.isVectorInstMem())
    {
        VectorMemIns++;
        //DPRINTF(VectorEngine,"Sending instruction %s to VMU, pc 0x%lx\n"
        //    ,insn.getName() , *(uint64_t*)&pc );
        vector_memory_unit->issue(*this,insn,dyn_insn, xc,src1,vtype,
            vl, done_callback);

        SumVL = SumVL.value() + vector_config->vector_length_in_bits(vl,vtype);
    }
    else if (insn.isVectorInstArith()) {

        SumVL = SumVL.value() + vector_config->vector_length_in_bits(vl,vtype);
        VectorArithmeticIns++;
        uint8_t lane_id_available = 0;
        for (int i=0 ; i< num_clusters ; i++) {
            if (!vector_lane[i]->isOccupied()) { lane_id_available = i; }
        }
        //DPRINTF(VectorEngine,"Sending instruction %s to cluster %d, pc 0x%lx\n",
        //    insn.getName(), lane_id_available , *(uint64_t*)&pc);
        vector_lane[lane_id_available]->issue(*this,insn,dyn_insn, xc, src1,
            vtype, vl,done_callback);
    } else {
        panic("Invalid Vector Instruction, insn=%#h\n", insn.machInst);
    }
}

Port &
VectorEngine::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "vector_reg_port")
        return VectorRegPorts[idx];
    else if (if_name == "vector_mem_port")
        return getVectorMemPort();
    else
        return getPort(if_name, idx);
}

VectorEngine::VectorMemPort::VectorMemPort(const std::string& name,
    VectorEngine* owner, uint8_t channels) :
    MasterPort(name, owner), owner(owner)
{
    //create the queues for each of the channels to the Vector Cache
    for (uint8_t i=0; i<channels; ++i) {
        laCachePktQs.push_back(std::deque<PacketPtr>());
    }
}

VectorEngine::VectorMemPort::~VectorMemPort()
{
}

VectorEngine::VectorMemPort::Tlb_Translation::Tlb_Translation(
    VectorEngine *owner):
    event(this, true), owner(owner)
{
}

VectorEngine::VectorMemPort::Tlb_Translation::~Tlb_Translation()
{
}

void
VectorEngine::VectorMemPort::Tlb_Translation::finish(const Fault &_fault,
    const RequestPtr &_req, ThreadContext *_tc, BaseTLB::Mode _mode)
{
    fault = _fault;
}

void
VectorEngine::VectorMemPort::Tlb_Translation::finish(const Fault _fault,
    uint64_t latency)
{
    fault = _fault;
}

void
VectorEngine::VectorMemPort::Tlb_Translation::markDelayed()
{
    panic("Tlb_Translation::markDelayed not implemented");
}

std::string
VectorEngine::VectorMemPort::Tlb_Translation::name()
{
    //TODO: IDK what to put here
    return "Tlb_Translation";
}

void
VectorEngine::VectorMemPort::Tlb_Translation::translated()
{
}

// submits a request for translation to the execCacheTlb
// MEMCPY data if present, so caller must deallocate it himself!
bool
VectorEngine::VectorMemPort::startTranslation(Addr addr, uint8_t *data,
    uint64_t size, BaseTLB::Mode mode, ThreadContext *tc, uint64_t req_id,
    uint8_t channel)
{
    Process * p = tc->getProcessPtr();
    Addr page1 = p->pTable->pageAlign(addr);
    Addr page2 = p->pTable->pageAlign(addr+size-1);
    assert(page1 == page2);

    //NOTE: need to make a buffer for reads so cache can write to it!
    uint8_t *ndata = new uint8_t[size];
    if (data != nullptr) {
        assert(mode == BaseTLB::Write);
        memcpy(ndata, data, size);
    } else {
      //put a pattern here for debugging
      memset(ndata, 'Z', size);
    }
    MemCmd cmd = (mode==BaseTLB::Write) ? MemCmd::WriteReq :
        MemCmd::ReadReq;

    //virtual address request constructor (copied the data port request)
    //const int asid = 0;
    const Addr pc = tc->instAddr();
    RequestPtr req = std::make_shared<Request>(addr, size, 0,
        owner->VectorCacheMasterId, pc, tc->contextId());

    BaseCPU *cpu = tc->getCpuPtr();

    req->taskId(cpu->taskId());

    //start translation
    Tlb_Translation *translation = new Tlb_Translation(owner);

    tc->getDTBPtr()->translateTiming(req, tc, translation , mode);

    if (translation->fault == NoFault){
        PacketPtr pkt = new VectorPacket(req, cmd, req_id, channel);
        pkt->dataDynamic(ndata);

        if (!sendTimingReq(pkt)) {
            laCachePktQs[channel].push_back(pkt);
            }

        delete translation;
        return true;
    }else{
        translation->fault->invoke(tc, NULL);
        delete translation;
        return false;
        }
}

bool
VectorEngine::VectorMemPort::sendTimingReadReq(Addr addr, uint64_t size,
    ThreadContext *tc, uint64_t req_id, uint8_t channel)
{
    return startTranslation(addr, nullptr, size, BaseTLB::Read, tc, req_id,
        channel);
}

bool
VectorEngine::VectorMemPort::sendTimingWriteReq(Addr addr,
    uint8_t *data, uint64_t size, ThreadContext *tc, uint64_t req_id,
    uint8_t channel)
{
    return startTranslation(addr, data, size, BaseTLB::Write, tc, req_id,
        channel);
}

bool
VectorEngine::VectorMemPort::recvTimingResp(PacketPtr pkt)
{
    VectorPacketPtr la_pkt = dynamic_cast<VectorPacketPtr>(pkt);
    assert(la_pkt != nullptr);
    owner->recvTimingResp(la_pkt);
    return true;
}

void
VectorEngine::VectorMemPort::recvReqRetry()
{
    //TODO: must be a better way to figure out which channel specified the
    //      retry
    for (auto& laCachePktQ : laCachePktQs) {
        //assert(laCachePktQ.size());
        while (laCachePktQ.size() && sendTimingReq(laCachePktQ.front())) {
            // memory system takes ownership of packet
            laCachePktQ.pop_front();
        }
    }
}

VectorEngine::VectorRegPort::VectorRegPort(const std::string& name,
    VectorEngine* owner, uint64_t channel) :
    MasterPort(name, owner), owner(owner), channel(channel)
{
}

VectorEngine::VectorRegPort::~VectorRegPort()
{
}

bool
VectorEngine::VectorRegPort::sendTimingReadReq(Addr addr, uint64_t size,
    uint64_t req_id)
{
    //physical addressing
    RequestPtr req = std::make_shared<Request>
        (addr,size,0,owner->VectorRegMasterIds[channel]);
    PacketPtr pkt = new VectorPacket(req, MemCmd::ReadReq, req_id);

    //make data for cache to put data into
    uint8_t *ndata = new uint8_t[size];
    memset(ndata, 'Z', size);
    pkt->dataDynamic(ndata);

    if (!sendTimingReq(pkt)) {
        //delete pkt->req;
        delete ndata;
        delete pkt;
        return false;
    }
    return true;
}

// memcpy of data, so the caller can delete it when it pleases
bool
VectorEngine::VectorRegPort::sendTimingWriteReq(Addr addr, uint8_t *data,
    uint64_t size, uint64_t req_id)
{
    //physical addressing
    RequestPtr req = std::make_shared<Request>
        (addr,size,0,owner->VectorRegMasterIds[channel]);
    VectorPacketPtr pkt = new VectorPacket(req, MemCmd::WriteReq, req_id);

    //make copy of data here
    uint8_t *ndata = new uint8_t[size];
    memcpy(ndata, data, size);
    pkt->dataDynamic(ndata);

    if (!sendTimingReq(pkt)){
        //delete pkt->req;
        delete ndata;
        delete pkt;
        return false;
    }
    return true;
}

bool
VectorEngine::VectorRegPort::recvTimingResp(PacketPtr pkt)
{
    VectorPacketPtr vector_pkt = dynamic_cast<VectorPacketPtr>(pkt);
    assert(vector_pkt != nullptr);
    owner->recvTimingResp(vector_pkt);
    return true;
}

void
VectorEngine::VectorRegPort::recvReqRetry()
{
    //do nothing, caller will manually resend request
}

void
VectorEngine::recvTimingResp(VectorPacketPtr vector_pkt)
{
    //find the associated request in the queue and associate with vector_pkt
    assert(vector_PendingReqQ.size());
    bool found = false;
    for (Vector_ReqState * pending : vector_PendingReqQ) {
        if (pending->getReqId() == vector_pkt->reqId){
            pending->setPacket(vector_pkt);
            found = true;
            break;
        }
    }
    assert(found);

    //commit each request in order they were issued
    while (vector_PendingReqQ.size() &&
        vector_PendingReqQ.front()->isMatched())
    {
        Vector_ReqState * pending = vector_PendingReqQ.front();
        vector_PendingReqQ.pop_front();
        pending->executeCallback();
        //NOTE: pending deletes packet. packet calls delete[] on the data
        delete pending;
    }
}


bool
VectorEngine::writeVectorMem(Addr addr, uint8_t *data, uint32_t size,
    ThreadContext *tc, uint8_t channel, std::function<void(void)> callback)
{
    DPRINTF(VectorEngine, "inside writeVectorMem\n");
    uint64_t id = (uniqueReqId++);
    Vector_ReqState *pending = new Vector_W_ReqState(id, callback);
    vector_PendingReqQ.push_back(pending);
    if (!vectormem_port.sendTimingWriteReq(addr, data, size, tc, id, channel)){
        delete vector_PendingReqQ.back();
        vector_PendingReqQ.pop_back();
        return false;
    }
    return true;
}

bool
VectorEngine::writeVectorReg(Addr addr, uint8_t *data,
    uint32_t size, uint8_t channel,
    std::function<void(void)> callback)
{
    DPRINTF(VectorEngine, "writeVectorReg got %d bytes to write at %#x\n"
        ,size, addr);
    uint64_t id = (uniqueReqId++);
    Vector_ReqState *pending = new Vector_W_ReqState(id, callback);
    vector_PendingReqQ.push_back(pending);
    if (!VectorRegPorts[channel].sendTimingWriteReq(addr,data,size,id)) {
        delete vector_PendingReqQ.back();
        vector_PendingReqQ.pop_back();
        return false;
    }
    return true;
}

bool
VectorEngine::readVectorMem(Addr addr, uint32_t size,
    ThreadContext *tc, uint8_t channel,
    std::function<void(uint8_t*,uint8_t)> callback)
{
    uint64_t id = (uniqueReqId++);
    Vector_ReqState *pending = new Vector_R_ReqState(id, callback);
    vector_PendingReqQ.push_back(pending);
    if (!vectormem_port.sendTimingReadReq(addr, size, tc, id, channel)) {
        delete vector_PendingReqQ.back();
        vector_PendingReqQ.pop_back();
        return false;
    }
    return true;
}

bool
VectorEngine::readVectorReg(Addr addr, uint32_t size,
    uint8_t channel,
    std::function<void(uint8_t*,uint8_t)> callback)
{
    uint64_t id = (uniqueReqId++);
    Vector_ReqState *pending = new Vector_R_ReqState(id,callback);
    vector_PendingReqQ.push_back(pending);
    if (!VectorRegPorts[channel].sendTimingReadReq(addr,size,id)) {
        delete vector_PendingReqQ.back();
        vector_PendingReqQ.pop_back();
        return false;
    }
    return true;
}

VectorEngine *
VectorEngineParams::create()
{
    return new VectorEngine(this);
}