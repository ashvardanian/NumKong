import std.stdio;
import numkong_d;

void main()
{
    nk_f32_t[] a = [1, 2, 3];
    nk_f32_t[] b = [4, 5, 6];
    nk_f64_t dot, kld;

    nk_configure_thread(nk_capabilities());

    nk_dot_f32(a.ptr, b.ptr, 3, &dot);
    writeln(i"Dot product of 2 vectors = $(dot)");
    nk_kld_f32(a.ptr, b.ptr, 3, &kld);
    writeln(i"Kullback Leibler distance of 2 vectors = $(kld)");
}
