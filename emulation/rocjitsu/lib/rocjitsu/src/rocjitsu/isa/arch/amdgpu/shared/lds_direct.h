// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

namespace rocjitsu::amdgpu {

/// RDNA3+ direct LDS loads broadcast one M0-selected value to whole active quads.
inline void execute_lds_direct_load(Wavefront &wf, uint32_t dst) {
  if (!wf.exec())
    return;
  const uint32_t type = (wf.m0() >> 16) & 7, address = wf.m0() & 0xffff;
  if ((address & 3) || type == 3 || type >= 6 || dst >= wf.num_vgprs()) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  uint32_t value = wf.lds().read32(wf.lds_base() + address);
  if (type != 2) {
    const uint32_t bits = type & 1 ? 16 : 8;
    value &= (1u << bits) - 1;
    if (type & 4)
      value = (value ^ (1u << (bits - 1))) - (1u << (bits - 1));
  }
  RegisterAccess regs(wf);
  for (uint32_t quad = 0; quad < wf.wf_size(); quad += 4)
    if (wf.exec() & (uint64_t{15} << quad))
      for (uint32_t lane = quad; lane < quad + 4; ++lane)
        regs.write_vgpr(wf.vgpr_alloc().base + dst, lane, value);
}

} // namespace rocjitsu::amdgpu
