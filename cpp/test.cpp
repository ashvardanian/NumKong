/**
 *  @file cpp/test.cpp
 *  @author Ash Vardanian
 *  @date November 26, 2026
 *  @brief The smoke test of the C++ SDK: a tensor and a vector round trip, a dot product and a
 *      distance, the scalar verbs, a cast and a packed matrix product, each against a known answer.
 *
 *  It links the archive a @c numkong::cpp consumer links, not the header-only engine, so every call
 *  dispatches through the compiled kernels. The precision suites in @c test/ hold each kernel to
 *  its reference on every capability; this one only checks that the public entry points are wired.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::printf`, `std::fprintf`

#include <source_location> // `std::source_location` - where a check failed
#include <vector>          // `std::vector` - the runs the concept layer takes

#include "numkong/numkong.hpp" // `nk::tensor`, `nk::vector`, `nk::dot`, `nk::cast`, `nk::dots_pack`

namespace nk = ashvardanian::numkong;

#pragma region Run

/** The run in progress, the one mutable global: the checks that failed in any case. */
struct test_run_t {
    std::size_t failed_checks = 0;
};

inline test_run_t test_run;

/** Holds when @p is_satisfied does; otherwise counts a failed check at the caller's line. */
inline bool expect_true(bool is_satisfied, std::source_location where = std::source_location::current()) noexcept {
    if (is_satisfied) return true;
    std::fprintf(stderr, "  check failed at %s:%u\n", where.file_name(), static_cast<unsigned>(where.line()));
    ++test_run.failed_checks;
    return false;
}

/** Runs one case and prints whether its checks held. */
inline void run_test_case(char const *label, void (*case_function)() noexcept) noexcept {
    std::size_t const failed_before = test_run.failed_checks;
    case_function();
    if (test_run.failed_checks == failed_before) std::printf("- %s ... ok\n", label);
    else std::fprintf(stderr, "- %s ... FAILED\n", label);
}

#pragma endregion Run

/** A rank-2 @c nk::tensor reads back what it was built from, by coordinate, offset and row. */
static void test_tensor_from() noexcept {
    auto [matrix, status] = nk::tensor<nk::f32_t>::from({{1, 2, 3}, {4, 5, 6}});
    if (!expect_true(nk::succeeded(status))) return;
    expect_true(matrix.rank() == 2 && matrix.extent(0) == 2 && matrix.extent(1) == 3);
    expect_true(float(matrix(1, 2)) == 6 && float(matrix[4]) == 5);

    nk::vector_view<nk::f32_t> const row = matrix.row(1).as_vector();
    expect_true(row.size() == 3 && float(row[0]) == 4 && float(row[2]) == 6);
}

/** An @c nk::vector keeps what is written through its pointer and iterates it back. */
static void test_vector_full() noexcept {
    auto [values, status] = nk::vector<nk::f32_t>::full(4, nk::f32_t {2});
    if (!expect_true(nk::succeeded(status))) return;
    expect_true(values.size() == 4);
    nk::f32_t *data = values.data();
    data[3] = nk::f32_t {7};
    float sum = 0;
    for (nk::f32_t const value : values) sum += float(value);
    expect_true(sum == 2 + 2 + 2 + 7);
}

/** ⟨(1,2,3),(4,5,6)⟩ is 32 through both the concept and the raw-pointer layers, and the squared
 *  distance between them 27. */
static void test_dot() noexcept {
    std::vector<nk::f32_t> const a {1, 2, 3}, b {4, 5, 6};
    auto [dot, status] = nk::dot<nk::f32_t>(a, b);
    expect_true(nk::succeeded(status) && double(dot) == 32);

    nk::f64_t raw {};
    expect_true(nk::succeeded(nk::dot(a.data(), b.data(), a.size(), &raw, nk::default_capabilities())));
    expect_true(double(raw) == double(dot));

    auto [distance, distance_status] = nk::sqeuclidean<nk::f32_t>(a, b);
    expect_true(nk::succeeded(distance_status) && double(distance) == 27);

    std::vector<nk::f32_t> const shorter {1, 2};
    expect_true(nk::dot<nk::f32_t>(a, shorter).status == nk::status_t::unexpected_dimensions_k);
}

/** The scalar types hold storage only, so converting, ordering, rooting and saturating go through
 *  the verbs: 1.5 survives @c f32 → @c f16 → @c f32, it sorts after zero, √4 is 2, and 100 + 100
 *  saturates to 127 in @c i8. */
static void test_scalar_verbs() noexcept {
    nk::f16_t const half = nk::cast<nk::f32_t, nk::f16_t>(1.5f);
    expect_true(nk::to_f32(half) == 1.5f);
    expect_true(nk::order(half, nk::f16_t::zero()) > 0 && nk::order(half, half) == 0);
    expect_true(float(nk::sqrt(nk::f32_t {4.0f})) == 2.0f);
    expect_true(int(nk::saturating_add(nk::i8_t {100}, nk::i8_t {100})) == 127);

    auto [narrow, status] = nk::cast<nk::f32_t, nk::e4m3_t>(nk::f32_t {-2.0f}, nk::default_capabilities(), nullptr);
    expect_true(nk::succeeded(status) && nk::to_f32(narrow) == -2.0f);
    expect_true(nk::sqrt(nk::f32_t {4.0f}, 0, nullptr).status == nk::status_t::missing_kernel_k);
}

/** Small integers and halves survive an @c f32 → @c bf16 → @c f32 round trip exactly. */
static void test_cast() noexcept {
    std::vector<nk::f32_t> const source {1, -2, 0.5f, 3, 0, 256};
    std::vector<nk::bf16_t> narrow(source.size());
    std::vector<nk::f32_t> wide(source.size());
    expect_true(nk::succeeded(nk::cast<nk::f32_t, nk::bf16_t>(source, narrow)));
    expect_true(nk::succeeded(nk::cast<nk::bf16_t, nk::f32_t>(narrow, wide)));
    for (std::size_t index = 0; index != source.size(); ++index)
        expect_true(float(wide[index]) == float(source[index]) && nk::to_f32(narrow[index]) == float(source[index]));
}

/** Ones times a packed matrix of twos over a depth of 4 is a matrix of eights, one row per
 *  left-hand row and one column per packed row. */
static void test_dots_pack() noexcept {
    auto [a, a_status] = nk::tensor<nk::f32_t>::full({2, 4}, nk::f32_t {1});
    auto [b, b_status] = nk::tensor<nk::f32_t>::full({3, 4}, nk::f32_t {2});
    if (!expect_true(nk::succeeded(a_status) && nk::succeeded(b_status))) return;
    auto [packed, packed_status] = nk::dots_pack<nk::f32_t>(b.as_matrix_view());
    if (!expect_true(nk::succeeded(packed_status))) return;
    expect_true(packed.rows() == 3 && packed.depth() == 4);

    auto [products, products_status] = nk::dots_packed<nk::f32_t>(a.as_matrix_view(), packed);
    if (!expect_true(nk::succeeded(products_status))) return;
    expect_true(products.extent(0) == 2 && products.extent(1) == 3);
    for (std::size_t row = 0; row != 2; ++row)
        for (std::size_t column = 0; column != 3; ++column) expect_true(double(products(row, column)) == 8);
}

int main() {
    std::printf("NumKong %d.%d.%d through the C++ SDK\n", NUMKONG_VERSION_MAJOR, NUMKONG_VERSION_MINOR,
                NUMKONG_VERSION_PATCH);
    run_test_case("nk::tensor::from", test_tensor_from);
    run_test_case("nk::vector::full", test_vector_full);
    run_test_case("nk::dot", test_dot);
    run_test_case("nk::cast<f32_t, f16_t>, nk::order, nk::sqrt", test_scalar_verbs);
    run_test_case("nk::cast", test_cast);
    run_test_case("nk::dots_pack", test_dots_pack);

    if (test_run.failed_checks != 0) {
        std::fprintf(stderr, "%zu check(s) failed\n", test_run.failed_checks);
        return 1;
    }
    std::printf("All checks passed\n");
    return 0;
}
