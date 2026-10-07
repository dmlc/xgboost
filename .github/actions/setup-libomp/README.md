# macOS OpenMP setup

This action installs Homebrew's `libomp` for the macOS Python wheel builds and
caches the installed library. Call it after checking out the repository:

```yaml
- uses: ./.github/actions/setup-libomp
```

## Why this exists

In October 2026, fresh CI runners repeatedly compiled dependencies during
`brew install libomp`. A representative Intel job spent about 19 minutes building
CMake and another 5 minutes building `libomp`, before building XGBoost. Across the
sample, the install-step medians were approximately 25 minutes on Intel and
14 minutes on ARM. See the [original job log][baseline] and [implementation PR][pr].

At that time, Homebrew no longer provided new Intel macOS bottles (prebuilt
packages), and the current `libomp` bottles did not cover our ARM `macos-14`
runner. These are time-specific conditions; consult Homebrew's current
[support policy][support] and [libomp formula][formula] when revisiting this action.

Caching downloads alone would still require compilation. Instead, we cache the
installed `Cellar/libomp` directory and its `opt/libomp` symlink. A hit skips the
entire install command, including installation of its build-time CMake dependency.
A miss still pays the original installation cost. The cache is saved immediately
after verifying the header and library exist, before wheel building begins.

## Maintenance

- Keys include architecture, full macOS version, and `libomp` version/revision.
  There are no broad fallback keys. Homebrew setup runs before the version query;
  auto-update is then disabled for the remaining job to avoid changing the
  formula between lookup and installation.
- CMake is a build dependency, not part of the restored library. If `libomp`
  gains non-system runtime dependencies, extend the cached installation or
  install those dependencies before using the restored library.
- Cache entries are immutable. Bump `macos-libomp-v1` when changing the cached
  layout or when a toolchain/runner change requires rebuilding an otherwise
  identically keyed installation. Evict a broken entry before retrying it.
- Transfer failures are nonfatal, but library verification and wheel builds are
  not. File-existence checks do not establish ABI compatibility; the wheel build
  and test jobs must still pass on both architectures.
- This action does not change wheel deployment targets or bundle `libomp` into
  the wheel. Preserve the existing OpenMP discovery and runtime-loading behavior
  when replacing it.

## When to remove it

Remove the caching wrapper when every remaining wheel-build runner can reliably
obtain a compatible prebuilt `libomp` quickly, or already includes it. This could
follow a runner upgrade, a move to another binary dependency source or a prepared
runner image, or retirement of the architectures needing source builds. An ARM
runner upgrade alone does not resolve the Intel source-build cost.

Before removing it, test installation on fresh runners with this cache bypassed
for each remaining architecture. Confirm the logs show no expensive dependency
source builds, then verify wheel builds, existing deployment targets, and runtime
tests. Replace the action call with the required lightweight installation step;
remove OpenMP setup entirely only if the build no longer needs external `libomp`.
A fast warm-cache run is not evidence that the workaround is obsolete.

[baseline]: https://github.com/dmlc/xgboost/actions/runs/37544783132/job/112532192434
[pr]: https://github.com/dmlc/xgboost/pull/12669
[support]: https://docs.brew.sh/Support-Tiers
[formula]: https://formulae.brew.sh/formula/libomp
