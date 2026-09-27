// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/transcendental_simd.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/amdgpu_exp.h"
#include "util/amdgpu_log.h"
#include "util/amdgpu_rcp.h"
#include "util/amdgpu_rsq.h"
#include "util/amdgpu_trig.h"
#include "util/simd_test_hooks.h"

#include <array>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <string>
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace {
using namespace rocjitsu;
using amdgpu::transcendental::evaluate_f32_simd;
using amdgpu::transcendental::F32Operation;
constexpr std::array operations{F32Operation::Log, F32Operation::Exp, F32Operation::Sin,
                                F32Operation::Cos, F32Operation::Rcp, F32Operation::Rsq};
constexpr std::array<uint32_t, 32> edges{
    0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x00800000, 0x33800000, 0xb3800000,
    0x39bfffff, 0x39c00000, 0x3d000001, 0x3d7fffff, 0x3e000000, 0x3e800001, 0x3f7bffff, 0x3f7c0000,
    0x3f7dffff, 0x3f7e0000, 0x3f7fffff, 0x3f800000, 0x3f800001, 0x3f810000, 0x3f820000, 0x3f83ffff,
    0x3f840000, 0xbf800000, 0x43000000, 0xc2fc0001, 0x7f800000, 0xff800000, 0x7f812345, 0xffc12345};
uint32_t reference(F32Operation op, uint32_t bits, unsigned denorm, bool quiet) {
  switch (op) {
  case F32Operation::Log:
    return util::detail::log::evaluate(bits, quiet);
  case F32Operation::Exp:
    return util::detail::exp::evaluate(bits, quiet);
  case F32Operation::Sin:
    return util::detail::trig::evaluate(bits, false, denorm, quiet);
  case F32Operation::Cos:
    return util::detail::trig::evaluate(bits, true, denorm, quiet);
  case F32Operation::Rcp:
    return std::bit_cast<uint32_t>(util::amdgpu_rcp_f32(std::bit_cast<float>(bits)));
  case F32Operation::Rsq:
    return std::bit_cast<uint32_t>(util::amdgpu_rsq_f32(std::bit_cast<float>(bits)));
  }
  return 0;
}

TEST(TranscendentalSimd, ExactMappingsAndInPlace) {
  std::mt19937 rng(0x89ab);
  for (unsigned batch = 0; batch < 64; ++batch) {
    std::array<uint32_t, 64> inputs{}, outputs{}, expected{};
    for (unsigned lane = 0; lane < inputs.size(); ++lane)
      inputs[lane] = batch == 0 ? edges[lane % edges.size()] : rng();
    for (auto operation : operations)
      for (unsigned denorm = 0; denorm < 4; ++denorm)
        for (bool quiet : {false, true})
          for (unsigned count : {0u, 1u, 4u, 7u, 8u, 9u, 32u, 63u, 64u}) {
            expected.fill(0xdeadbeef);
            outputs = expected;
            for (unsigned lane = 0; lane < count; ++lane)
              expected[lane] = reference(operation, inputs[lane], denorm, quiet);
            evaluate_f32_simd(operation, inputs.data(), outputs.data(), count, denorm, quiet);
            ASSERT_EQ(outputs, expected) << "batch " << batch << " op " << unsigned(operation);
            outputs = inputs;
            evaluate_f32_simd(operation, outputs.data(), outputs.data(), count, denorm, quiet);
            for (unsigned lane = count; lane < inputs.size(); ++lane)
              expected[lane] = inputs[lane];
            ASSERT_EQ(outputs, expected);
          }
  }
}

TEST(TranscendentalSimd, PreservesHostEnvironment) {
  std::fenv_t saved{};
  ASSERT_EQ(std::feholdexcept(&saved), 0);
  const int saved_errno = errno;
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    std::fesetround(rounding);
    std::feraiseexcept(FE_DIVBYZERO | FE_INEXACT);
    for (auto op : operations) {
      std::fenv_t before{}, after{};
      std::fegetenv(&before);
      std::array<uint32_t, 32> output{};
      errno = 123;
      evaluate_f32_simd(op, edges.data(), output.data(), output.size(), 3, false);
      const int observed_errno = errno;
      std::fegetenv(&after);
      EXPECT_EQ(observed_errno, 123);
      EXPECT_EQ(std::memcmp(&before, &after, sizeof(before)), 0);
    }
  }
  std::fesetenv(&saved);
  errno = saved_errno;
}

struct ForceScalarGuard {
  bool original = util::force_scalar();
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(original); }
};

TEST(TranscendentalSimd, DecodedInstructionsPreserveMasksAliasesAndBroadcasts) {
  ForceScalarGuard guard;
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3,
                    ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA1,
                    ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4,
                    ROCJITSU_CODE_ARCH_CDNA5}) {
    const bool gcn = arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
                     arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
    for (unsigned width : {32u, 64u}) {
      if ((gcn && width == 32) || (arch == ROCJITSU_CODE_ARCH_CDNA5 && width == 64))
        continue;
      SCOPED_TRACE(testing::Message() << "arch " << arch << " width " << width);
      amdgpu::GpuMemory memory("transcendental_memory");
      amdgpu::L2Cache cache("transcendental_cache");
      cache.set_backing_memory(&memory);
      amdgpu::ComputeUnitCore::Config cfg{};
      cfg.arch = arch;
      cfg.num_wf_slots = 1;
      cfg.sgprs_per_wf = 106;
      cfg.vgprs_per_wf = 256;
      cfg.lds_size_kb = 64;
      auto cu = amdgpu::ComputeUnitCore::create("transcendental_cu", cfg, &memory, &cache);
      auto decoder = Decoder::create(arch);
      auto *wave = cu->dispatch_wf(0, 0, 106, 256, width);
      ASSERT_NE(wave, nullptr);
      const uint32_t vgpr = wave->vgpr_alloc().base;
      const uint64_t full = width == 64 ? ~uint64_t{0} : 0xffffffffu;
      for (unsigned op = 0; op < operations.size(); ++op)
        for (bool e64 : {false, true})
          for (bool alias : {false, true})
            for (bool broadcast : {false, true})
              for (unsigned modifiers : {0u, 1u, 2u, 3u}) {
                if (!e64 && modifiers)
                  continue;
                const uint32_t dst = alias ? 0 : 6;
                // Same independently assembled opcodes as the FP MODE fixtures.
                const uint32_t e32_rdna[]{0x7e004f00, 0x7e004b00, 0x7e006b00,
                                          0x7e006d00, 0x7e005500, 0x7e005d00};
                const uint32_t e32_gcn[]{0x7e004300, 0x7e004100, 0x7e005300,
                                         0x7e005500, 0x7e004500, 0x7e004900};
                const uint32_t e64_rdna[]{0xd5a70000, 0xd5a50000, 0xd5b50000,
                                          0xd5b60000, 0xd5aa0000, 0xd5ae0000};
                const uint32_t e64_gcn[]{0xd1610000, 0xd1600000, 0xd1690000,
                                         0xd16a0000, 0xd1620000, 0xd1640000};
                std::array<uint32_t, 3> words{};
                if (e64) {
                  words[0] = (gcn ? e64_gcn[op] : e64_rdna[op]) | dst;
                  words[1] = gcn ? 0x100 : 0x02010100;
                  if (modifiers & 1)
                    words[0] |= 1u << 8;
                  if (modifiers & 2)
                    words[1] |= 1u << 29;
                  if (broadcast)
                    words[1] &= ~0x1ffu;
                } else {
                  words[0] = (gcn ? e32_gcn[op] : e32_rdna[op]) | (dst << 17);
                  if (broadcast)
                    words[0] &= ~0x1ffu;
                }
                auto decoded = decoder->decode(words.data());
                ASSERT_FALSE(decoded.failed());
                auto instruction = std::move(decoded).value();
                const char *names[]{"v_log_f32", "v_exp_f32", "v_sin_f32",
                                    "v_cos_f32", "v_rcp_f32", "v_rsq_f32"};
                EXPECT_EQ(instruction->mnemonic(), std::string(names[op]) + (e64 ? "" : "_e32"));
                for (unsigned denorm : {0u, 3u})
                  for (uint64_t mask :
                       {uint64_t{0}, uint64_t{1}, uint64_t{0x80}, uint64_t{0xf}, uint64_t{0xff},
                        uint64_t{0x80000001}, uint64_t{0x5555555555555555}, full}) {
                    std::array<uint32_t, 64> expected{}, actual{};
                    for (bool scalar : {true, false}) {
                      util::set_force_scalar_for_testing(scalar);
                      wave->set_mode_raw(192u | (denorm << 4));
                      wave->set_exec(mask & full);
                      amdgpu::RegisterAccess(*wave).write_sgpr(wave->sgpr_alloc().base, 0x3f800001);
                      for (unsigned lane = 0; lane < width; ++lane) {
                        cu->write_vgpr(vgpr + 6, lane, 0xdeadbeef);
                        cu->write_vgpr(vgpr, lane, edges[lane % edges.size()]);
                      }
#if defined(__x86_64__)
                      std::fenv_t original{}, before{}, after{};
                      const int original_errno = errno;
                      unsigned expected_csr = 0;
                      if (op < 2) {
                        // Check the integer LOG/EXP probe and its scalar fallback
                        // with independent SSE/x87 rounding and enabled traps.
                        std::feholdexcept(&original);
                        constexpr int rounds[]{FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO};
                        std::fesetround(rounds[modifiers]);
                        expected_csr = (_mm_getcsr() & ~0xe07fu) | 0x4000u;
                        if (denorm)
                          expected_csr |= 0x8040u;
                        if (mask & 1) {
                          expected_csr &= ~0x1f80u;
                          unsigned short control;
                          asm volatile("fnstcw %0" : "=m"(control));
                          control &= ~0x3fu;
                          asm volatile("fldcw %0" : : "m"(control));
                        } else {
                          expected_csr |= 0x24u;
                        }
                        _mm_setcsr(expected_csr);
                        std::fegetenv(&before);
                        errno = EILSEQ;
                      }
#endif
                      const bool succeeded =
                          cu->execute_instruction(instruction.get(), *wave).succeeded();
#if defined(__x86_64__)
                      if (op < 2) {
                        const int observed_errno = errno;
                        const unsigned observed_csr = _mm_getcsr();
                        std::fegetenv(&after);
                        // Restore before invoking assertions or formatting.
                        std::fesetenv(&original);
                        errno = original_errno;
                        EXPECT_EQ(observed_errno, EILSEQ);
                        EXPECT_EQ(observed_csr, expected_csr);
                        EXPECT_EQ(std::memcmp(&before, &after, sizeof(before)), 0);
                      }
#endif
                      ASSERT_TRUE(succeeded);
                      auto &output = scalar ? expected : actual;
                      for (unsigned lane = 0; lane < width; ++lane)
                        output[lane] = cu->read_vgpr(vgpr + dst, lane);
                    }
                    ASSERT_EQ(actual, expected)
                        << "arch " << arch << " op " << op << " width " << width << " modifiers "
                        << modifiers << " EXEC " << mask;
                  }
              }
      wave->halt();
    }
  }
}
} // namespace
