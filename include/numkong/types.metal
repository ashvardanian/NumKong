/**
 *  @file include/numkong/types.metal
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Integer arithmetic helpers shared by Metal kernels.
 *
 *  @sa include/numkong/types.h
 */
namespace nk {

/** Divides rounding up without overflowing the numerator. */
template <typename number_type_, typename divisor_type_>
constexpr number_type_ divide_round_up(number_type_ number, divisor_type_ divisor) {
    return number / divisor + (number % divisor != 0);
}

/** Rounds up to a multiple of @p divisor; the rounded value must fit the result type. */
template <typename number_type_, typename divisor_type_>
constexpr number_type_ round_up_to_multiple(number_type_ number, divisor_type_ divisor) {
    return divide_round_up(number, divisor) * divisor;
}

} // namespace nk
