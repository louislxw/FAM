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
	int data; // int data
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

void framing(
	stream<single_fixed_t >& i_in,
	stream<single_fixed_t >& q_in,
	stream<single_fixed_t >& i_out,
	stream<single_fixed_t >& q_out,
	const unsigned P // 32
);

void windowing(
    stream<single_fixed_t >& i_in,
    stream<single_fixed_t >& q_in,
	stream<single_fixed_t >& i_out,
	stream<single_fixed_t >& q_out,
	const unsigned P // 32
);

void fft_ip_256(
	stream<single_fixed_t >& i_in,
	stream<single_fixed_t >& q_in,
	stream<single_fixed_t1 >& i_out,
	stream<single_fixed_t1 >& q_out,
	const unsigned P // 32
);

void bit_reverser_256(
	stream<single_fixed_t >& i_in,
	stream<single_fixed_t >& q_in,
    stream<single_fixed_t >& i_out,
    stream<single_fixed_t >& q_out,
	const unsigned P // 32
);

void fft_256(
	stream<single_fixed_t >& i_in,
	stream<single_fixed_t >& q_in,
	stream<single_fixed_t1 >& i_out,
	stream<single_fixed_t1 >& q_out,
	const unsigned num_ffts // 32
);

void down_conversion(
	stream<single_fixed_t1 >& i_in,
	stream<single_fixed_t1 >& q_in,
	stream<single_fixed_t2 >& i_out,
	stream<single_fixed_t2 >& q_out,
	const unsigned P // 32
);

void preprocess(
    stream<single_fixed_t >& i_in,
	stream<single_fixed_t >& q_in,
	stream<single_fixed_t2 >& i_out,
	stream<single_fixed_t2 >& q_out
);
// This is the end of preprocess!!!

void conjugate_mult(
	stream<single_fixed_t2 >& i_in,
	stream<single_fixed_t2 >& q_in,
	stream<single_fixed_t3 >& i_out,
	stream<single_fixed_t3 >& q_out,
	const unsigned Np // 256
);

void bit_reverser_32(
    stream<single_fixed_t3 >& i_in,
    stream<single_fixed_t3 >& q_in,
    stream<single_fixed_t3 >& i_out,
    stream<single_fixed_t3 >& q_out,
	const unsigned num_rows // 65536
);

void fft_32(
    stream<single_fixed_t3 >& i_in,
    stream<single_fixed_t3 >& q_in,
    stream<single_fixed_t4 >& i_out,
	stream<single_fixed_t4 >& q_out,
	const unsigned num_ffts // 256*256 = 65536
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
    stream<single_fixed_t5 >& alpha_out
);
// This is the end of SCD matrix!!!

void model(
	stream<single_fixed_t >& i_in,
	stream<single_fixed_t >& q_in,
	stream<single_fixed_t5 >& alpha_out
);

void model_wrapper(
	stream<ap_int<16> >& axis_i_in,
	stream<ap_int<16> >& axis_q_in,
	AXIS_OUT &axis_i_out,
	AXIS_OUT &axis_q_out
);
/*
void model_wrapper(
	stream<single_fixed_t >& i_in,
	stream<single_fixed_t >& q_in,
	AXIS_OUT &axis_i_out,
	AXIS_OUT &axis_q_out
);*/
// This is the end of fixed-point modules!!!

/********** Floating-point modules **********/
void framing_float(
	stream<float >& i_in,
	stream<float >& q_in,
	stream<single_fixed_t >& i_out, //
	stream<single_fixed_t >& q_out, //
	const unsigned P // 32
);

void bit_reverser_256_float(
	stream<single_fixed_t >& i_in, //
	stream<single_fixed_t >& q_in, //
    stream<single_fixed_t >& i_out, //
    stream<single_fixed_t >& q_out, //
	const unsigned P // 32
);

void fft_256_float(
	stream<single_fixed_t >& i_in, //
	stream<single_fixed_t >& q_in, //
	stream<single_fixed_t1 >& i_out, //
	stream<single_fixed_t1 >& q_out, //
	const unsigned num_ffts // 32
);

void down_conversion_float(
	stream<single_fixed_t1 >& i_in, //
	stream<single_fixed_t1 >& q_in, //
	stream<single_fixed_t2 >& i_out, //
	stream<single_fixed_t2 >& q_out, //
	const unsigned P // 32
);

void preprocess_float(
    stream<float >& i_in,
	stream<float >& q_in,
	stream<single_fixed_t2 >& i_out,
	stream<single_fixed_t2 >& q_out
);

void conjugate_mult_float(
	stream<single_fixed_t2 >& i_in,
	stream<single_fixed_t2 >& q_in,
	stream<single_fixed_t3 >& i_out,
	stream<single_fixed_t3 >& q_out,
	const unsigned Np // 256
);

void bit_reverser_32_float(
    stream<single_fixed_t3 >& i_in,
    stream<single_fixed_t3 >& q_in,
    stream<single_fixed_t3 >& i_out,
    stream<single_fixed_t3 >& q_out,
	const unsigned num_rows // 65536
);

void fft_32_float(
    stream<single_fixed_t3 >& i_in,
    stream<single_fixed_t3 >& q_in,
    stream<single_fixed_t4 >& i_out,
	stream<single_fixed_t4 >& q_out,
	const unsigned num_ffts // 256*256 = 65536
);

void alpha_profile_float(
	stream<single_fixed_t4 >& i_in,
	stream<single_fixed_t4 >& q_in,
	stream<single_fixed_t5 >& alpha_out,
	const unsigned R // 256
);

void scd_matrix_float(
    stream<single_fixed_t2 >& i_in,
    stream<single_fixed_t2 >& q_in,
    stream<single_fixed_t5 >& alpha_out
);

void model_float(
	stream<float >& i_in,
	stream<float >& q_in,
	stream<single_fixed_t5 >& alpha_out
);
// This is the end of floating-point modules!!!

int main(int argc, char **argv) {
	printf("Hello, World!\n");

/********** AXI Stream input/output **********/
	axis_in stream_i_in, stream_q_in;
	axis_out stream_i_out, stream_q_out, stream_alpha_out;
//	axis_in_float stream_i_in_float, stream_q_in_float;
	axis_out_float stream_i_out_float, stream_q_out_float, stream_alpha_out_float;

/********** Fixed-point input/output **********/
	// top level in/out stream
	stream<ap_int<16> > i_in("i_in_stream");
	stream<ap_int<16> > q_in("q_in_stream");
	// stream<single_fixed_t > i_in("i_in_stream");
	// stream<single_fixed_t > q_in("q_in_stream");
	stream<single_fixed_t5 > alpha_out("alpha_out_stream");

	// preprocess stream
	stream<single_fixed_t2 > i_pre_out("i_preprocess_out_stream");
	stream<single_fixed_t2 > q_pre_out("q_preprocess_out_stream");

	// preprocess stream (sub blocks)
	stream<single_fixed_t > i_frame_out("i_frame_out_stream");
	stream<single_fixed_t > q_frame_out("q_frame_out_stream");
	stream<single_fixed_t > i_window_out("i_window_out_stream");
	stream<single_fixed_t > q_window_out("q_window_out_stream");
	stream<single_fixed_t > i_rev_256_out("i_rev_256_out_stream");
	stream<single_fixed_t > q_rev_256_out("q_rev_256_out_stream");
	stream<single_fixed_t1 > i_fft_256_out("i_fft_256_out_stream");
	stream<single_fixed_t1 > q_fft_256_out("q_fft_256_out_stream");

	// scd_matrix stream (sub blocks)
	stream<single_fixed_t3 > i_conjugate_out("i_conjugate_out_stream");
	stream<single_fixed_t3 > q_conjugate_out("q_conjugate_out_stream");
	stream<single_fixed_t3 > i_rev_32_out("i_rev_32_out_stream");
	stream<single_fixed_t3 > q_rev_32_out("q_rev_32_out_stream");
	stream<single_fixed_t4 > i_fft_32_out("i_fft_32_out_stream");
	stream<single_fixed_t4 > q_fft_32_out("q_fft_32_out_stream");

/********** Floating-point input/output **********/
	// top level in/out stream
	stream<float > i_in_float("i_in_float_stream");
	stream<float > q_in_float("q_in_float_stream");
	stream<single_fixed_t5 > alpha_out_float("alpha_out_float_stream");

	stream<single_fixed_t2 > i_pre_out_float("i_pre_out_float_stream"); //
	stream<single_fixed_t2 > q_pre_out_float("i_pre_out_float_stream"); //

	// preprocess stream (sub blocks) (float)
	stream<single_fixed_t > i_frame_out_float("i_frame_out_float_stream"); //
	stream<single_fixed_t > q_frame_out_float("q_frame_out_float_stream"); //
	stream<single_fixed_t > i_rev_256_out_float("i_rev_256_out_float_stream"); //
	stream<single_fixed_t > q_rev_256_out_float("q_rev_256_out_float_stream"); //
	stream<single_fixed_t1 > i_fft_256_out_float("i_fft_256_out_float_stream"); //
	stream<single_fixed_t1 > q_fft_256_out_float("q_fft_256_out_float_stream"); //

	// scd_matrix stream (sub blocks) (float)
	stream<single_fixed_t3 > i_conjugate_out_float("i_conjugate_out_float_stream");
	stream<single_fixed_t3 > q_conjugate_out_float("q_conjugate_out_float_stream");
	stream<single_fixed_t3 > i_rev_32_out_float("i_rev_32_out_float_stream");
	stream<single_fixed_t3 > q_rev_32_out_float("q_rev_32_out_float_stream");
	stream<single_fixed_t4 > i_fft_32_out_float("i_fft_32_out_float_stream");
	stream<single_fixed_t4 > q_fft_32_out_float("q_fft_32_out_float_stream");

	int Np = 256;
	int L = Np/4; // 64
	int P = 32;

	int in_size = 8192; // (P-1)*L + Np; // 2240
	int out_size = 8192; // Np*P/2; // 4096

	for (int i = 0; i < in_size; i++) { // 2241
		stream_i_in.data = (i%9); // single_fixed_t (i%4/10.0)
		stream_q_in.data = (i%4);
		i_in.write(stream_i_in.data);
		q_in.write(stream_q_in.data);
//		printf("i_in[%d] = %d, q_in[%d] = %d\n", i-1, (int)stream_i_in.data, i-1, (int)stream_q_in.data);
	}

/*	framing(i_in, q_in, i_frame_out, q_frame_out, 32); // Step 1 (Verified!)

	windowing(i_frame_out, q_frame_out, i_window_out, q_window_out, 32); // step 1.2

	fft_ip_256(i_window_out, q_window_out, i_fft_256_out, q_fft_256_out, 32);

//	bit_reverser_256(i_window_out, q_window_out, i_rev_256_out, q_rev_256_out, 32); // Step 2.1 (Verified!)

//	fft_256(i_rev_256_out, q_rev_256_out, i_fft_256_out, q_fft_256_out, 32); // Step 2.2 (Verified!)

	down_conversion(i_fft_256_out, q_fft_256_out, i_pre_out, q_pre_out, 32); // Step 3 (Verified!)

	framing_float(i_in_float, q_in_float, i_frame_out_float, q_frame_out_float, 32); // Step 1 (Verified!)

	bit_reverser_256_float(i_frame_out_float, q_frame_out_float, i_rev_256_out_float, q_rev_256_out_float, 32); // Step 2.1 (Verified!)

	fft_256_float(i_rev_256_out_float, q_rev_256_out_float, i_fft_256_out_float, q_fft_256_out_float, 32); // Step 2.2 (Verified!)

	down_conversion_float(i_fft_256_out_float, q_fft_256_out_float, i_pre_out_float, q_pre_out_float, 32); // Step 3 (Verified!)
*/

//	preprocess(i_in, q_in, i_pre_out, q_pre_out); // Step 1-3

//	preprocess_float(i_in_float, q_in_float, i_pre_out_float, q_pre_out_float); // Step 1-3

	AXIS_OUT axis_i_out("axis_i_out_stream");
	AXIS_OUT axis_q_out("axis_q_out_stream");

	model_wrapper(i_in, q_in, axis_i_out, axis_q_out);

	float fixed_square = 0, float_square = 0, error = 0, error_sum = 0, accuracy = 0, dif_square = 0;
	float i_error = 0, q_error = 0, i_error_max = 0, q_error_max = 0, RMSE = 0;
	for (int i = 0; i < P; i++) {
    	for (int j = 0; j < Np; j++) {
    		stream_i_out = axis_i_out.read();
    		stream_q_out = axis_q_out.read();
    		if (i == 0) {
    			printf("i_out[%d][%d] = %x, q_out[%d][%d] = %x, TLAST = %d\n", i, j, (stream_i_out.data), i, j, (stream_q_out.data), (unsigned)stream_i_out.last);
    		}
    	}
	}


/*	conjugate_mult(i_pre_out, q_pre_out, i_conjugate_out, q_conjugate_out, 256); // Step 4.1 (Verified!)

	bit_reverser_32(i_conjugate_out, q_conjugate_out, i_rev_32_out, q_rev_32_out, 65536); // Step 4.2.1 (Verified!)

	fft_32(i_rev_32_out, q_rev_32_out, i_fft_32_out, q_fft_32_out, 65536); // Step 4.2.2 (Verified!)

	conjugate_mult_float(i_pre_out_float, q_pre_out_float, i_conjugate_out_float, q_conjugate_out_float, 256); // Step 4.1 (Verified!)

	bit_reverser_32_float(i_conjugate_out_float, q_conjugate_out_float, i_rev_32_out_float, q_rev_32_out_float, 65536); // Step 4.2.1 (Verified!)

	fft_32_float(i_rev_32_out_float, q_rev_32_out_float, i_fft_32_out_float, q_fft_32_out_float, 65536); // Step 4.2.2 (Verified!)
*/
/*	float error = 0, i_error = 0, q_error = 0, i_error_max = 0, q_error_max = 0, RMSE = 0;
	for (int i = 0; i < Np; i++) {
		for (int j = 0; j < Np; j++) {
			for (int k = 0; k < P; k++) {
				stream_i_out.data = i_fft_32_out.read();
				stream_q_out.data = q_fft_32_out.read();
				stream_i_out_float.data = i_fft_32_out_float.read();
				stream_q_out_float.data = q_fft_32_out_float.read();
				if (i == 0 && k == 0) {
					printf("i_out[%d][%d] = %f, q_out[%d][%d] = %f\n", i*256+j, k, (float)stream_i_out.data, i*256+j, k, (float)stream_q_out.data);
					printf("i_out_f[%d][%d] = %f, q_out_f[%d][%d] = %f\n", i*256+j, k, (float)stream_i_out_float.data, i*256+j, k, (float)stream_q_out_float.data);
				}
				i_error = abs(float (stream_i_out.data) - stream_i_out_float.data);
				q_error = abs(float (stream_q_out.data) - stream_q_out_float.data);
				error += i_error*i_error + q_error*q_error;
				if (i_error > i_error_max) {
					i_error_max = i_error;
				}
				if (q_error > q_error_max) {
					q_error_max = q_error;
				}
			}
		}
    }
	RMSE = sqrt(error)/(Np*Np*P);
	printf("i_error_max = %f,  q_error_max = %f\n", (float)i_error_max,  (float)q_error_max);
	printf("RMSE = %f\n", (float)RMSE);
*/

/*	alpha_profile(i_fft_32_out, q_fft_32_out, alpha_out, 256); // Step 4.3

	alpha_profile_float(i_fft_32_out_float, q_fft_32_out_float, alpha_out_float, 256); // Step 4.3

//	scd_matrix(i_pre_out, q_pre_out, alpha_out);

//	scd_matrix_float(i_pre_out_float, q_pre_out_float, alpha_out_float);

	float error = 0, alpha_error = 0, alpha_error_max = 0, RMSE = 0;
	for (int i = 0; i < out_size; i++) {
    		stream_alpha_out.data = alpha_out.read();
    		stream_alpha_out_float.data = alpha_out_float.read();
    		printf("alpha_out[%d] = %f\n", i, (float)stream_alpha_out.data);
    		printf("alpha_out_float[%d] = %f\n", i, (float)stream_alpha_out_float.data);
    		alpha_error = abs(float (stream_alpha_out.data) - stream_alpha_out_float.data);
    		error += alpha_error*alpha_error;
    		if (alpha_error > alpha_error_max) {
    			alpha_error_max = alpha_error;
    		}
	}
	RMSE = sqrt(error)/out_size;
	printf("alpha_error_max = %f\n", (float)alpha_error_max);
	printf("RMSE = %f\n", (float)RMSE);
*/

	return 0;
}
