# Proposal: empty-Scratch / empty-TempStorage convention

## The problem

We currently express "this class needs no smem state" by defining `TempStorage` (or `ScratchStorage`) as `struct {}` (or, equivalently, as a `using` alias to some empty struct). Several places in the code already use `cuda::std::is_empty_v<T>` to detect this and pick a smaller layout (e.g. `partition_storage_layout_for_t` chooses between the persistent and non-persistent shape based on `is_empty_v<TempStorage>`).

That works *until* we wrap the storage in `cub::Uninitialized<>`. Wrapping is what we just adopted across `BlockPartition` / `BlockFilter` / `multi_source_data_source` to satisfy the "no dynamic init in `__shared__`" rule for unions with non-trivial alternatives. But:

```cpp
sizeof(Uninitialized<empty_struct>) == 1   // not 0, has DeviceWord storage[N]
is_empty_v<Uninitialized<empty_struct>> == false
```

So once an inner empty type passes through `Uninitialized<>`, the empty signal is **lost** to the consumer:

- A composition (`multi_source_data_source<direct, direct>`, where both children have empty `ScratchStorage`) ends up with a non-empty `ScratchStorage` after wrapping.
- A consumer can no longer write `if constexpr (is_empty_v<typename Inner::TempStorage>) { /* skip __syncthreads */ }` because the trait returns `false` even when the inner storage is empty.

## The use cases

Two concrete use cases drive the need for a robust empty-detection trait:

### Use case 1: skip `__syncthreads()` between aliased consumers

The agent unions multiple consumers' smem footprints to compress total smem. Between transitioning from consumer A's writes to consumer B's reads on aliased memory, an explicit `__syncthreads()` is needed so A's writes are visible — *unless* A wrote nothing (empty TempStorage). Today the agent always emits the barrier defensively; if it could detect "A's TempStorage is empty" it could elide it.

### Use case 2: propagate empty up through compositions

`multi_source_data_source<A, B>::ScratchStorage` is structurally empty whenever both children are. We'd like the composition to surface this, both for further nesting (a top-level container holding a `multi_source<direct, direct>` should see "this scratch is empty") and to skip the storage allocation entirely.

## Proposal

A two-piece convention: a canonical marker, plus a smart wrapper.

### 1. Canonical empty-storage marker — reuse `cub::NullType`

CUB already overloads `NullType` as the "empty / sentinel" alias-type-marker (e.g. agents write `using agent_value_data_source_scratch_t = conditional_t<keys_only, NullType, value_source_t::ScratchStorage>` today). Formalise this for storage:

> **Convention:** if a class has no per-tile or per-segment smem state, it should publish that as `using TempStorage = ::cub::NullType;` (or `using ScratchStorage = ::cub::NullType;`) — a *type alias*, not a `struct {}`.

`NullType` is empty (`is_empty_v<NullType>` is `true`) and trivially copyable (defaulted default ctor; the templated `explicit NullType(const T&)` is not the default ctor, so it doesn't break trivial-default-init). It is therefore safe to:
- name it from compile-time traits,
- declare it as `__shared__` (no dynamic init issue),
- embed it as a member of any `__shared__`-resident composite (zero or 1-byte cost depending on EBO).

Alternative considered: a brand-new `empty_storage_t` marker. Rejected because (a) `NullType` is already canonical for "no T here" across CUB and (b) this would be a third sentinel to remember. The downside is mild semantic overload: `NullType` now means both "no value type" and "no storage type". Acceptable — the contexts don't collide.

### 2. Detection trait — `is_empty_storage_v<T>`

```cpp
template <typename T>
inline constexpr bool is_empty_storage_v =
    ::cuda::std::is_same_v<T, ::cub::NullType>
 || ::cuda::std::is_empty_v<T>;
```

The `is_empty_v` clause keeps existing `struct {}`-style declarations working without migration. Encouraged style for new code is the alias to `NullType`; existing `struct {}` carry the same meaning to consumers.

### 3. Smart wrapper — `shared_storage_for_t<T>`

```cpp
template <typename T>
using shared_storage_for_t = ::cuda::std::conditional_t<
    is_empty_storage_v<T>,
    T,                          // empty -> pass through; do NOT wrap
    ::cub::Uninitialized<T>>;   // non-empty -> wrap so __shared__ is happy
```

Replaces every `cub::Uninitialized<inner>` we introduced in the recent refactor that *might* receive an empty inner. For empty inners, it leaves the type alone — preserving the empty signal for downstream consumers. For non-empty inners, it wraps as before.

A matching access helper hides the `Alias()` plumbing:

```cpp
// Returns a reference to the underlying storage T regardless of whether the
// storage was wrapped in `Uninitialized<>` (non-empty case) or pass-through
// (empty case).
template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE T& alias_shared_storage(shared_storage_for_t<T>& s)
{
  if constexpr (is_empty_storage_v<T>) {
    return s;                  // s is already T (or NullType)
  } else {
    return s.Alias();          // unwrap Uninitialized<T>
  }
}
```

So a consumer that today writes `buffer.Alias().value_load.x = ...` would write `alias_shared_storage<inner_t>(buffer).value_load.x = ...`, and the same code compiles whether `buffer` is the wrapped or pass-through form.

### 4. Composition propagation

For a class whose storage is *conditionally* empty based on its children, write the storage type as a `conditional_t` over the composite trait. Example for `multi_source_data_source<A, B>::ScratchStorage`:

```cpp
private:
  static constexpr bool _scratch_is_empty =
       is_empty_storage_v<typename A::ScratchStorage>
    && is_empty_storage_v<typename B::ScratchStorage>;

  union _scratch_inner_full {
    typename A::ScratchStorage a;
    typename B::ScratchStorage b;
  };

public:
  using ScratchStorage = ::cuda::std::conditional_t<
      _scratch_is_empty,
      ::cub::NullType,
      shared_storage_for_t<_scratch_inner_full>>;
```

When both children are empty, the multi_source publishes `NullType` upward. Otherwise it publishes the wrapped union. The trait at the next level up sees the right answer in either case.

The same pattern applies anywhere a composite's storage is the union of its children's storages (`block_partition_atomics`, `block_filter_atomics`, the `partition_storage_layout` helper, etc.).

### 5. The use case-1 idiom

A consumer that previously did

```cpp
__syncthreads();   // sync between aliased phases
B::process(buffer);
```

becomes

```cpp
if constexpr (!is_empty_storage_v<typename A::TempStorage>) {
  __syncthreads();
}
B::process(buffer);
```

…with similar `if constexpr` guards anywhere a barrier protects ordering against a (possibly-empty) writer. The agent decides at compile time, the SASS sees no branch.

## Migration — what would change

The convention is opt-in. Existing `struct {}`-style empty storages keep working through the `is_empty_v` arm of the trait. The migration amounts to:

1. **Define the marker, trait, and helpers** in a new small header (e.g. `cub/detail/topk/empty_storage.cuh` to start; promote to `cub/util_type.cuh` if the convention proves out).

2. **Switch `struct TempStorage {}` / `struct ScratchStorage {}` to type aliases** in storage-defining classes:
   ```cpp
   using TempStorage   = ::cub::NullType;
   using ScratchStorage = ::cub::NullType;
   ```
   Affected classes (sweep, top-k subtree only):
   - `direct_data_source` (both members).
   - `sync_block_load_data_source` `TempStorage` (currently empty).
   - `block_partition_atomics`, `block_partition_staged`, `block_partition_shared_mem`, `block_filter_atomics`, `block_filter_staged`, `block_filter_shared_mem` `TempStorage`.
   - The accumulating sister classes' `ScratchStorage` where empty.

3. **Replace `Uninitialized<inner>` with `shared_storage_for_t<inner>`** at every smem-allocation site we just touched in the refactor:
   - `multi_source_data_source::ScratchStorage`.
   - `block_partition_atomics::ScratchStorage` (the new value-load embedding).
   - `block_filter_atomics::ScratchStorage` (same).
   - Both `phase_t` unions in `block_partition_staged` / `block_partition_shared_mem` / `block_filter_staged` / `block_filter_shared_mem`.
   - Both layouts in `partition_storage_layout`.

4. **Replace `.Alias()` calls with `alias_shared_storage<inner>(...)`** where the inner type may be empty. This is mostly the access sites I already updated in the recent refactor.

5. **Add propagation `conditional_t` to compositions** so the upward type stays empty when all children are empty (`multi_source_data_source`, the `_ScratchStorage` wrapper in atomics, the inner unions in staged / shared_mem, etc.).

6. **Sprinkle `if constexpr (!is_empty_storage_v<T::TempStorage>) { __syncthreads(); }` guards** at the use-case-1 sites the agent identifies.

## Edge cases / open questions

1. **`Uninitialized<empty>` is currently size 1.** Today, after my refactor, `Uninitialized<NullType>` is what `multi_source_data_source<direct, direct>::ScratchStorage` ends up being, and that 1 byte is what we see in our snapshots. Under the new convention, that drops to `NullType` directly — which is still `sizeof == 1` due to C++ EBO rules unless `[[no_unique_address]]` kicks in for its enclosing class. So in absolute byte terms, the new convention is **not necessarily smaller** in the typical case; the win is in detection (is_empty_storage_v becomes truthful) and the freedom to elide barriers / unused setup code.

2. **`__shared__ NullType x;`** — should compile (NullType is trivially default-constructible). I haven't tested, would want a CI sanity test to confirm.

3. **`is_empty_storage_v` permissiveness vs strictness.** I'm proposing the permissive form (alias-or-empty-struct) so existing code keeps working. The downside is anyone who *accidentally* declares an empty struct gets it treated as the "skip barrier" marker. I think that's actually fine — it correctly reflects what the type *is* — but it's worth flagging.

4. **Mixing wrappers across header boundaries.** If module A exposes `using TempStorage = NullType;` and module B accidentally wraps in `Uninitialized<TempStorage>` directly (instead of `shared_storage_for_t`), the empty signal is lost again. Convention discipline is the only safeguard. A `[[deprecated]]`-style trip wire on `Uninitialized<NullType>` would catch this in lint, but feels heavy.

5. **What about `Uninitialized<some_empty_struct>` written by hand?** Under the permissive trait, the inner is empty, so `shared_storage_for_t` would elide the wrap. But if a caller went around `shared_storage_for_t` and wrote `Uninitialized<some_empty_struct>` directly, they'd get the 1-byte wrapper and lose detection. Again, convention discipline.

6. **Should the helpers live in the topk subtree or in `cub::` proper?** I'd start in `cub/detail/topk/empty_storage.cuh` (close to the consumers, no API exposure), then promote to `cub/util_type.cuh` next to `Uninitialized` and `NullType` once the convention is proven.

## Naming alternatives to bikeshed

- Marker: `NullType` (proposed) vs `empty_storage_t` (cleaner, but new sentinel).
- Trait: `is_empty_storage_v` (proposed) vs `has_no_storage_v`, `is_storage_empty_v`, `storage_is_empty_v`.
- Wrapper: `shared_storage_for_t` (proposed) vs `smem_storage_for_t`, `aliased_storage_for_t`, `wrapped_storage_for_t`.
- Access helper: `alias_shared_storage` vs `as_storage`, `unwrap_storage`.

## What this would look like in `multi_source_data_source` (before / after sketch)

**After my recent refactor (current state on the side branch):**

```cpp
template <typename SourceA, typename SourceB, typename OffsetT = ...>
class multi_source_data_source {
public:
  union _ScratchStorageInner {
    typename SourceA::ScratchStorage a;
    typename SourceB::ScratchStorage b;
  };
  struct ScratchStorage : ::cub::Uninitialized<_ScratchStorageInner> {};
  // ...
};
```

**Under the proposed convention:**

```cpp
template <typename SourceA, typename SourceB, typename OffsetT = ...>
class multi_source_data_source {
private:
  static constexpr bool _scratch_is_empty =
       is_empty_storage_v<typename SourceA::ScratchStorage>
    && is_empty_storage_v<typename SourceB::ScratchStorage>;

  union _scratch_inner_full {
    typename SourceA::ScratchStorage a;
    typename SourceB::ScratchStorage b;
  };

public:
  using ScratchStorage = ::cuda::std::conditional_t<
      _scratch_is_empty,
      ::cub::NullType,
      shared_storage_for_t<_scratch_inner_full>>;

  // ...
};
```

For our typical `multi_source<direct, direct>` configuration both children have empty scratch → `ScratchStorage = NullType`. Downstream consumers (e.g. `block_partition_atomics::_ScratchStorage`) then see `is_empty_storage_v<...> == true` and either drop the embedded value-load member entirely (back to `struct ScratchStorage = NullType`) or guard barriers around it.

## Summary

- **Convention**: `using TempStorage / ScratchStorage = NullType;` for empty cases.
- **Trait**: `is_empty_storage_v<T>` recognises the alias *and* legacy empty structs.
- **Helper**: `shared_storage_for_t<T>` wraps in `Uninitialized<>` only when needed.
- **Composition**: each class declares its storage as `conditional_t` over its children's emptiness so the empty signal propagates upward.
- **Consumer use**: `if constexpr (!is_empty_storage_v<T::TempStorage>) __syncthreads();` (and similar setup elision).
- **Migration**: opt-in, backward-compatible with existing `struct {}` style.

The net effect is twofold: (a) compile-time emptiness becomes a first-class trait we can actually rely on across class boundaries, even after smem wrapping, and (b) the agent gains a mechanical way to elide setup work that's only needed when storage is non-empty.
