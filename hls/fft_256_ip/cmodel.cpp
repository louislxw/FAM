#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"
#include "hls_stream.h"

#include "exp_table.h"
#include "window.h"
#include "types.h"
#include "fft_top.h"

using namespace hls;

/*** Define AXI-Stream structures ***/
struct axis_out{
	int data; // ap_int<32> data
	bool last;
};
typedef hls::stream<axis_out> AXIS_OUT;

typedef union {
	int i; // int i
	float f; // float f;
} fconvert;

void dummy_proc_fe(
	bool direction,
	config_t* config,
	cmpxDataIn in[FFT_LENGTH],
	cmpxDataIn out[FFT_LENGTH]
) {
	int i;
    config->setDir(direction);
    config->setSch(0xAA); // [2 2 2 2]
    for (i=0; i< FFT_LENGTH; i++)
    	out[i] = in[i];
}

void dummy_proc_be(
    status_t* status_in,
    bool* ovflo,
	cmpxDataOut in[FFT_LENGTH],
	cmpxDataOut out[FFT_LENGTH])
{
    int i;
    for (i=0; i< FFT_LENGTH; i++)
        out[i] = in[i];
    *ovflo = status_in->getOvflo() & 0x1;
}

void fft_top(
	bool direction,
	cmpxDataIn in[FFT_LENGTH],
	cmpxDataOut out[FFT_LENGTH],
	bool* ovflo
) {
//#pragma HLS interface ap_hs port=direction
//#pragma HLS interface ap_fifo depth=1 port=ovflo
#pragma HLS interface ap_fifo depth=FFT_LENGTH port=in,out
#pragma HLS data_pack variable=in
#pragma HLS data_pack variable=out
#pragma HLS dataflow

	cmpxDataIn xn[FFT_LENGTH];
	cmpxDataOut xk[FFT_LENGTH];
	config_t fft_config;
	status_t fft_status;

#pragma HLS interface ap_fifo port=fft_config
#pragma HLS data_pack variable=fft_config

	// FFT in configuration
	dummy_proc_fe(direction, &fft_config, in, xn);
	// FFT IP
	hls::fft<config1>(xn, xk, &fft_status, &fft_config);
	// FFT out configuration
	dummy_proc_be(&fft_status, ovflo, xk, out);
}

void fft_ip_256(
	stream<single_fixed_t >& i_in,
	stream<single_fixed_t >& q_in,
	stream<single_fixed_t1 >& i_out,
	stream<single_fixed_t1 >& q_out,
	const unsigned P // 32
) {
#pragma HLS dataflow
	for (int p = 0; p < P; p++) {
		int Np = 256;
		static cmpxDataIn fft_in[FFT_LENGTH];
		static cmpxDataOut fft_out[FFT_LENGTH];
		bool fft_direction = 1;
		bool fft_ovflo;
//#pragma HLS PIPELINE II=1
//#pragma HLS interface ap_hs port=fft_direction
//#pragma HLS interface ap_fifo depth=1 port=fft_ovflo
#pragma HLS interface ap_fifo depth=FFT_LENGTH port=fft_in,fft_out
#pragma HLS data_pack variable=fft_in
#pragma HLS data_pack variable=fft_out
		for (int i = 0; i < Np; i++) {
#pragma HLS UNROLL
			single_fixed_t re, im;
			re = i_in.read();
			im = q_in.read();
			fft_in[i] = cmpxDataIn(re, im);
		}

		fft_top(fft_direction, fft_in, fft_out, &fft_ovflo);

		for (int i = 0; i < Np; i++) {
#pragma HLS UNROLL
			cmpxDataOut out_temp[FFT_LENGTH];
			out_temp[i] = fft_out[i];
			i_out.write(out_temp[i].real() ); // << 3
			q_out.write(out_temp[i].imag() ); // << 3
//			printf("i_fft_out[%d] = %f,\t q_fft_out[%d] = %f\n", i, (float)xk[i].real(), i, (float)xk[i].imag());
		}
	}
}

// Step 2 (256-pt FFT using Xilinx IP core)
/*void fft_ip_256(
	stream<single_fixed_t >& i_in,
	stream<single_fixed_t >& q_in,
	stream<single_fixed_t1 >& i_out,
	stream<single_fixed_t1 >& q_out,
	const unsigned P // 32
) {
	static cmpxDataIn xn[FFT_LENGTH];
	static cmpxDataOut xk[FFT_LENGTH];
	config_t fft_config;
	status_t fft_status;
	// Forward FFT
//	fft_config.setDir(1);
//	fft_config.setNfft(8);
	// Set FFT length to 256 => log2(256) = 8
//	fft_config.setSch(0xAA); // [2 2 2 2]
// #pragma HLS dataflow
#pragma HLS interface ap_fifo depth=256 port=xn,xk
#pragma HLS interface ap_fifo port=fft_config
#pragma HLS data_pack variable=xn
#pragma HLS data_pack variable=xk
#pragma HLS data_pack variable=fft_config

	for (int p = 0; p < P; p++) {
#pragma HLS PIPELINE II=1
		int Np = 256;
		cmpxDataIn fft_in[FFT_LENGTH];
		for (int i = 0; i < Np; i++) {
#pragma HLS UNROLL
			single_fixed_t re, im;
			re = i_in.read();
			im = q_in.read();
			fft_in[i] = cmpxDataIn(re, im);
		}
		// FFT configuration
		dummy_proc_fe(&fft_config, fft_in, xn);
		// FFT IP
		hls::fft<config1>(xn, xk, &fft_status, &fft_config);

		for (int i = 0; i < Np; i++) {
#pragma HLS UNROLL
			cmpxDataOut out_temp[FFT_LENGTH];
			out_temp[i] = xk[i];
			i_out.write(out_temp[i].real() ); // << 3
			q_out.write(out_temp[i].imag() ); // << 3
//			printf("i_fft_out[%d] = %f,\t q_fft_out[%d] = %f\n", i, (float)xk[i].real(), i, (float)xk[i].imag());
		}
	}
}*/

// AXI Stream wrapper
void model_wrapper(
	stream<ap_int<16> >& axis_i_in,
	stream<ap_int<16> >& axis_q_in,
	AXIS_OUT &axis_i_out,
	AXIS_OUT &axis_q_out
){
// #pragma HLS INTERFACE s_axilite port=batch_size bundle=ctrl
// #pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=axis_i_in
#pragma HLS INTERFACE axis port=axis_q_in
#pragma HLS INTERFACE axis port=axis_i_out
#pragma HLS INTERFACE axis port=axis_q_out
#pragma HLS DATAFLOW

	const int in_size = 8192; // P*Np = 32*256 = 8192
	const int out_size = 8192; // P*Np = 32*256 = 8192
	stream<single_fixed_t > i_in;
	stream<single_fixed_t > q_in;
	stream<single_fixed_t > i_out;
	stream<single_fixed_t > q_out;
	stream<single_fixed_t > i_frame_out("i_frame_out_stream");
	stream<single_fixed_t > q_frame_out("q_frame_out_stream");
	stream<single_fixed_t > i_window_out("i_window_out_stream");
	stream<single_fixed_t > q_window_out("q_window_out_stream");

	for (int j = 0; j < in_size; j++) {
// #pragma HLS unroll factor=256
		fconvert i_conv, q_conv;
		i_conv.f = (axis_i_in.read())/10.0;
		q_conv.f = axis_q_in.read();
		i_in.write(i_conv.f);
		q_in.write(q_conv.f);
	}

	fft_ip_256(i_in, q_in, i_out, q_out, 32);

	for (int j = 0; j < out_size; j++) {
// #pragma HLS unroll factor=256
		axis_out i_tmp_out;
		axis_out q_tmp_out;

		fconvert i_conv, q_conv;
		i_conv.f = i_out.read();
		q_conv.f = q_out.read();
		i_tmp_out.data = (i_conv.i); // i_conv.i;
		q_tmp_out.data = (q_conv.i); // q_conv.i;

		if(j == (out_size-1)) {
			i_tmp_out.last = 1;
			q_tmp_out.last = 1;
		}
		else {
			i_tmp_out.last = 0;
			q_tmp_out.last = 0;
		}
//  		if (j>=0 && j<256) {
//  			printf("i_fixed[%d] = %f, i_hex[%d] = %x\n", j, i_conv.f, j, i_tmp_out.data);
//   		}

		axis_i_out.write(i_tmp_out);
		axis_q_out.write(q_tmp_out);
	}
}
