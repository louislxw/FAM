#include "hls_stream.h"
#include "ap_int.h"
#include "ap_fixed.h"
#include "math.h"

#include "exp_table.h"
#include "window.h"
#include "types.h"
#include "hls_fft.h"

using namespace hls;

/*** Define AXI-Stream structures ***/
struct axis_in{
	int data; // single_fixed_t data
	bool last;
};
struct axis_out{
	int data; // single_fixed_t data
	bool last;
};
typedef hls::stream<axis_in> AXIS_IN;
typedef hls::stream<axis_out> AXIS_OUT;

struct axis_in_float{
	float data;
	bool last;
};
struct axis_out_float{
	float data;
	bool last;
};

void conjugate_mult(
	stream<single_fixed_t2 >& i_in,
	stream<single_fixed_t2 >& q_in,
	stream<single_fixed_t3 >& i_out,
	stream<single_fixed_t3 >& q_out
);

void bit_reverser_32(
    stream<single_fixed_t3 >& i_in,
    stream<single_fixed_t3 >& q_in,
    stream<single_fixed_t3 >& i_out,
    stream<single_fixed_t3 >& q_out
);

void fft_32(
    stream<single_fixed_t3 >& i_in,
    stream<single_fixed_t3 >& q_in,
    stream<single_fixed_t4 >& i_out,
	stream<single_fixed_t4 >& q_out
);

void alpha_profile(
	stream<single_fixed_t4 >& i_in,
	stream<single_fixed_t4 >& q_in,
	stream<single_fixed_t5 >& alpha_out,
	const unsigned R // 256
);

void scd_matrix(
    stream<single_fixed_t2 >& i_in,
    stream<single_fixed_t2 >& q_in,
    stream<single_fixed_t4 >& i_out,
	stream<single_fixed_t4 >& q_out
);
// This is the end of SCD matrix!!!

void scd_pe_array(
	stream<single_fixed_t2 >& i_in,
	stream<single_fixed_t2 >& q_in,
    stream<single_fixed_t4 >& i_out,
	stream<single_fixed_t4 >& q_out
);

void model_wrapper(
	stream<ap_int<16> >& axis_i_in,
	stream<ap_int<16> >& axis_q_in,
	AXIS_OUT &axis_i_out,
	AXIS_OUT &axis_q_out
);
// This is the end of fixed-point modules!!!

int main(int argc, char **argv) {

	printf("Hello, World!\n");

/********** AXI Stream input/output **********/
	axis_in stream_i_in, stream_q_in;
	axis_out stream_i_out, stream_q_out, stream_alpha_out;
	AXIS_OUT axis_i_out("axis_i_out_stream");
	AXIS_OUT axis_q_out("axis_q_out_stream");
/********** Fixed-point input/output **********/
	stream<ap_int<16> > i_in("i_in_stream");
	stream<ap_int<16> > q_in("q_in_stream");

	int Np = 256;
	int L = Np/4; // 64
	int P = 32;

	int in_size = 32; // P*L = 2048 or (P-1)*L + Np = 2240
	int out_size = 32; // 4096

	for (int i = 0; i < in_size; i++) { // 2241
		stream_i_in.data = (i%10); // single_fixed_t (i%4/10.0)
		stream_q_in.data = 0;
		i_in.write(stream_i_in.data);
		q_in.write(stream_q_in.data);
//		printf("i_in[%d] = %d, q_in[%d] = %d\n", i, (int)stream_i_in.data, i, (int)stream_q_in.data);
	}

	model_wrapper(i_in, q_in, axis_i_out, axis_q_out);

    for (int i = 0; i < out_size; i++) {
		stream_i_out = axis_i_out.read();
		stream_q_out = axis_q_out.read();
		printf("i_out[%d] = %x, q_out[%d] = %x, TLAST = %d\n", i, stream_i_out.data, i, stream_q_out.data, (unsigned)stream_i_out.last);
    }

	return 0;
}
