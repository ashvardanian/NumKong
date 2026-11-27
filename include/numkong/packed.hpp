/**
 *  @file include/numkong/packed.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief The owner of many packed point clouds, the operand every many-to-many shape kernel takes.
 *
 *  @c nk::shapes_pack in the @c numkong::cpp headers packs one through the dispatching kernels, and
 *  @c shapes_packed::make through any pair of callables, like one capability's kernels.
 */
#ifndef NUMKONG_PACKED_HPP
#define NUMKONG_PACKED_HPP

#include <cstddef> // `std::size_t`

#include <memory>      // `std::allocator_traits`
#include <span>        // `std::span`
#include <type_traits> // `std::is_same_v`
#include <utility>     // `std::exchange`

#include "numkong/memory.hpp" // `aligned_allocator`
#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/** The point types of shape packs: @c f64_t and @c f32_t compute in @c f64_t, and @c f16_t in
 *  @c f32_t. */
template <typename in_type_>
concept a_shapes_dtype = std::is_same_v<in_type_, f64_t> || std::is_same_v<in_type_, f32_t> ||
                         std::is_same_v<in_type_, f16_t>;

/**
 *  @brief Owning, move-only set of packed clouds, the operand every many-to-many kernel takes.
 *
 *  One allocation holds the table of pack pointers, then the packs back to back, each on an
 *  @c nk_default_alignment_k boundary, so @c data() passes straight to every shape kernel.
 */
template <a_shapes_dtype value_type_, typename allocator_type_ = aligned_allocator<char>>
class shapes_packed {
    using alloc_traits = std::allocator_traits<allocator_type_>;

    char *data_ = nullptr;
    std::size_t size_bytes_ = 0;
    std::size_t count_ = 0;
    std::size_t max_points_ = 0;
    [[no_unique_address]] allocator_type_ alloc_;

    std::size_t table_bytes_() const noexcept {
        return nk_size_round_up_to_multiple_(count_ * sizeof(void const *), nk_default_alignment_k);
    }

  public:
    /** The allocator of the packs' one block, which uses-allocator construction looks up. */
    using allocator_type = allocator_type_;

    shapes_packed() noexcept = default;
    explicit shapes_packed(allocator_type_ const &alloc) noexcept : alloc_(alloc) {}

    ~shapes_packed() noexcept {
        if (data_) alloc_traits::deallocate(alloc_, data_, size_bytes_);
    }

    shapes_packed(shapes_packed &&o) noexcept
        : data_(std::exchange(o.data_, nullptr)), size_bytes_(std::exchange(o.size_bytes_, 0)),
          count_(std::exchange(o.count_, 0)), max_points_(std::exchange(o.max_points_, 0)),
          alloc_(std::move(o.alloc_)) {}

    shapes_packed &operator=(shapes_packed &&o) noexcept {
        if (this != &o) {
            if (data_) alloc_traits::deallocate(alloc_, data_, size_bytes_);
            if constexpr (alloc_traits::propagate_on_container_move_assignment::value) alloc_ = std::move(o.alloc_);
            data_ = std::exchange(o.data_, nullptr);
            size_bytes_ = std::exchange(o.size_bytes_, 0);
            count_ = std::exchange(o.count_, 0);
            max_points_ = std::exchange(o.max_points_, 0);
        }
        return *this;
    }

    shapes_packed(shapes_packed const &) = delete;
    shapes_packed &operator=(shapes_packed const &) = delete;

    /**
     *  @brief Packs every cloud of @p clouds, each @b [points,3] interleaved, through one pair of
     *      callables, such as one capability's @c nk_shapes_pack_size_* and @c nk_shapes_pack_*.
     *  @param[in] clouds The point clouds to pack, one pack per cloud.
     *  @param[in] alloc Source of the packs' blocks.
     *  @param[in] pack_size Measures one pack as <tt>pack_size(points, &bytes)</tt>; pair it with @p pack.
     *  @param[in] pack Packs one cloud as <tt>pack(raw_points, points, packed, stream)</tt>, with a null
     *      stream; a normalized one packs like @c nk::shapes_pack_normalized.
     *  @return The packs.
     *
     *  @par Statuses:
     *  - @c unexpected_dimensions_k A cloud's length is not a multiple of 3.
     *  - The failure @p pack_size or @p pack reports.
     */
    template <typename pack_size_type_, typename pack_type_>
    static expected<shapes_packed> make(std::span<std::span<value_type_ const> const> clouds, allocator_type_ alloc,
                                        pack_size_type_ pack_size, pack_type_ pack) noexcept {
        shapes_packed packs(alloc);
        packs.count_ = clouds.size();
        std::size_t total = packs.table_bytes_();
        for (std::span<value_type_ const> cloud : clouds) {
            if (cloud.size() % 3) return {shapes_packed(alloc), status_t::unexpected_dimensions_k};
            nk_size_t bytes = 0;
            if (status_t const sized = static_cast<status_t>(pack_size(cloud.size() / 3, &bytes)); failed(sized))
                return {shapes_packed(alloc), sized};
            total += nk_size_round_up_to_multiple_(bytes, nk_default_alignment_k);
            if (packs.max_points_ < cloud.size() / 3) packs.max_points_ = cloud.size() / 3;
        }
        if (!total) return {std::move(packs), status_t::success_k};
        packs.size_bytes_ = total;
        packs.data_ = alloc_traits::allocate(packs.alloc_, total);
        if (!packs.data_) return {shapes_packed(alloc), status_t::bad_alloc_k};

        void const **table = reinterpret_cast<void const **>(packs.data_);
        std::size_t offset = packs.table_bytes_();
        for (std::size_t index = 0; index != clouds.size(); ++index) {
            std::size_t const points = clouds[index].size() / 3;
            auto const *cloud = reinterpret_cast<typename value_type_::raw_t const *>(clouds[index].data());
            table[index] = packs.data_ + offset;
            if (status_t const packed = static_cast<status_t>(pack(cloud, points, packs.data_ + offset, nullptr));
                failed(packed))
                return {shapes_packed(alloc), packed};
            nk_size_t bytes = 0;
            static_cast<void>(pack_size(points, &bytes));
            offset += nk_size_round_up_to_multiple_(bytes, nk_default_alignment_k);
        }
        return {std::move(packs), status_t::success_k};
    }

    /** The table of pack pointers the kernels take. */
    void const *const *data() const noexcept { return reinterpret_cast<void const *const *>(data_); }

    /** The pack of cloud @p index. */
    void const *operator[](std::size_t index) const noexcept { return data()[index]; }

    /** The number of packed clouds. */
    std::size_t size() const noexcept { return count_; }

    /** Read through @c nk::shapes_packed_shape on the pack. */
    std::size_t points(std::size_t index) const = delete;

    /** The largest cloud, which sizes the transport workspaces. */
    std::size_t max_points() const noexcept { return max_points_; }

    bool empty() const noexcept { return data_ == nullptr; }
    std::size_t size_bytes() const noexcept { return size_bytes_; }

    /** The allocator the packs came from, so a replacement can be packed from the same one. */
    allocator_type get_allocator() const noexcept { return alloc_; }
};

} // namespace ashvardanian::numkong

#endif // NUMKONG_PACKED_HPP
