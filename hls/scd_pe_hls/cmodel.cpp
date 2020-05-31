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

typedef union{
	int i;
	float f;
} fconvert;

/********** SCD Matrix **********/
// Step 4.1
void conjugate_mult(
	stream<single_fixed_t2 >& i_in,
	stream<single_fixed_t2 >& q_in,
	stream<single_fixed_t3 >& i_out,
	stream<single_fixed_t3 >& q_out
){
#pragma HLS DATAFLOW
/*** Two Memories for I/Q stream ***/
	single_fixed_t2 i_mem[32];
	single_fixed_t2 q_mem[32];

	for (int i = 0; i < 32; i++) {
#pragma HLS UNROLL
		i_mem[i] = i_in.read();
		q_mem[i] = q_in.read();
	}
/*** Y = X[i] * conjugate(X[j]) Complex Mult requires 4 DSPs. ***/
	for (int i = 0; i < 32; i++) {
//#pragma HLS UNROLL
//		single_fixed_t i_tmp = i_mem[k][j] * i_mem[i][j] + q_mem[k][j] * q_mem[i][j];
//		single_fixed_t q_tmp = q_mem[k][j] * i_mem[i][j] - i_mem[k][j] * q_mem[i][j];
		double_fixed_t2 ii_tmp = i_mem[i] * i_mem[i];
		double_fixed_t2 qq_tmp = q_mem[i] * q_mem[i];
		double_fixed_t2 iq_tmp = i_mem[i] * q_mem[i];
		double_fixed_t2 qi_tmp = q_mem[i] * i_mem[i];
		double_fixed_t2 i_tmp = (ii_tmp + qq_tmp) >> 1; //
		double_fixed_t2 q_tmp = (qi_tmp - iq_tmp) >> 1; //
		i_out.write(i_tmp);
		q_out.write(q_tmp);
	}
}

// Xilinx FFT IP core
/*
void dummy_proc_fe(
	bool direction,
	config_t* config,
	cmpxDataIn in[FFT_LENGTH],
	cmpxDataIn out[FFT_LENGTH]
) {
	int i;
    config->setDir(direction);
    config->setSch(0x1A); // [1 2 2]
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

void fft_ip_32(
	stream<single_fixed_t3 >& i_in,
	stream<single_fixed_t3 >& q_in,
	stream<single_fixed_t4 >& i_out,
	stream<single_fixed_t4 >& q_out
) {
#pragma HLS dataflow
	int Np = 32;
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
//		printf("i_fft_out[%d] = %f,\t q_fft_out[%d] = %f\n", i, (float)xk[i].real(), i, (float)xk[i].imag());
	}
}
*/

// Step 4.2.1 (Bit reversal for 32-point FFT)
void bit_reverser_32(
    stream<single_fixed_t3 >& i_in,
    stream<single_fixed_t3 >& q_in,
    stream<single_fixed_t3 >& i_out,
    stream<single_fixed_t3 >& q_out
){
#pragma HLS DATAFLOW
	const unsigned indices[32] = {
		0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,
		1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31
	};
	single_fixed_t3 ibuf[32];
	single_fixed_t3 qbuf[32];

	for (int i = 0; i < 32; i++) {
#pragma HLS UNROLL
		ibuf[i] = i_in.read();
		qbuf[i] = q_in.read();
	}
	for (int i = 0; i < 32; i++) {
#pragma HLS UNROLL
		i_out.write(ibuf[indices[i]]);
		q_out.write(qbuf[indices[i]]);
	}
}

// Step 4.2.2 (32-point FFT in time-lag & FFT-shift)
void fft_32(
    stream<single_fixed_t3 >& i_in,
    stream<single_fixed_t3 >& q_in,
    stream<single_fixed_t4 >& i_out,
	stream<single_fixed_t4 >& q_out
){
//#pragma HLS DATAFLOW
//#pragma HLS PIPELINE II=1
		double_fixed_t3 ibuf[32];
		double_fixed_t3 qbuf[32];
		for (int i = 0; i < 32; i++) {
#pragma HLS UNROLL
			// read the input data
			ibuf[i] = i_in.read();
			qbuf[i] = q_in.read();
		}

		// size = 2
		int halfsize = 1;
		int tablestep = 16;

		for (int i = 0; i < 32; i += 2) {
//#pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
//#pragma HLS PIPELINE II=1
				double_fixed_t3 temp_ii, temp_qq, temp_iq, temp_qi;
#pragma HLS RESOURCE variable=temp_ii core=HMul_nodsp
#pragma HLS RESOURCE variable=temp_qq core=HMul_nodsp
#pragma HLS RESOURCE variable=temp_iq core=HMul_nodsp
#pragma HLS RESOURCE variable=temp_qi core=HMul_nodsp
				 temp_ii = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_i[k];
				 temp_qq = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_i = temp_ii - temp_qq;
				 temp_iq = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_q[k];
				 temp_qi = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_q = temp_iq + temp_qi;
//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_32_i[k]) - (qbuf[j+halfsize] * exp_table_32_q[k])) >> 14;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_32_q[k]) + (qbuf[j+halfsize] * exp_table_32_i[k])) >> 14;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) >> 1; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) >> 1; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) >> 1; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) >> 1; // >> 1
				k += tablestep;
			}
		}

		// size = 4
		halfsize = 2;
		tablestep = 8;

		for (int i = 0; i < 32; i += 4) {
//#pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
//#pragma HLS PIPELINE II=1
				double_fixed_t3 temp_ii = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_qq = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_i = temp_ii - temp_qq;
				double_fixed_t3 temp_iq = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_qi = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_q = temp_iq + temp_qi;
//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_32_i[k]) - (qbuf[j+halfsize] * exp_table_32_q[k])) >> 14;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_32_q[k]) + (qbuf[j+halfsize] * exp_table_32_i[k])) >> 14;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) >> 1; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) >> 1; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) >> 1; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) >> 1; // >> 1
				k += tablestep;
			}
		}

		// size = 8
		halfsize = 4;
		tablestep = 4;

		for (int i = 0; i < 32; i += 8) {
//#pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
//#pragma HLS PIPELINE II=1
				double_fixed_t3 temp_ii = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_qq = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_i = temp_ii - temp_qq;
				double_fixed_t3 temp_iq = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_qi = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_q = temp_iq + temp_qi;
//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_32_i[k]) - (qbuf[j+halfsize] * exp_table_32_q[k])) >> 14;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_32_q[k]) + (qbuf[j+halfsize] * exp_table_32_i[k])) >> 14;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) >> 1; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) >> 1; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) >> 1; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) >> 1; // >> 1
				k += tablestep;
			}
		}

		// size = 16
		halfsize = 8;
		tablestep = 2;

		for (int i = 0; i < 32; i += 16) {
//#pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
//#pragma HLS PIPELINE II=1
				double_fixed_t3 temp_ii = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_qq = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_i = temp_ii - temp_qq;
				double_fixed_t3 temp_iq = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_qi = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_q = temp_iq + temp_qi;
//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_32_i[k]) - (qbuf[j+halfsize] * exp_table_32_q[k])) >> 14;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_32_q[k]) + (qbuf[j+halfsize] * exp_table_32_i[k])) >> 14;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) >> 1; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) >> 1; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) >> 1; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) >> 1; // >> 1
				k += tablestep;
			}
		}

		// size = 32
		halfsize = 16;
		tablestep = 1;

		for (int i = 0; i < 32; i += 32) {
//#pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
//#pragma HLS PIPELINE II=1
				double_fixed_t3 temp_ii = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_qq = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_i = temp_ii - temp_qq;
				double_fixed_t3 temp_iq = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_qi = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_q = temp_iq + temp_qi;
//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_32_i[k]) - (qbuf[j+halfsize] * exp_table_32_q[k])) >> 14;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_32_q[k]) + (qbuf[j+halfsize] * exp_table_32_i[k])) >> 14;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) >> 1; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) >> 1; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) >> 1; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) >> 1; // >> 1
				k += tablestep;
			}
		}
		// Directly write out the FFT output
//		for (int i = 0; i < 32; i++) {
//#pragma HLS UNROLL
//			i_out.write(ibuf[i]);
//			q_out.write(qbuf[i]);
//		}
		// FFT-shift (shifting the zero-frequency component to the center of the array)
		for (int i = 0; i < 32; i++) {
#pragma HLS UNROLL
			if (i < 16) {
				i_out.write(ibuf[i+16]);
				q_out.write(qbuf[i+16]);
			}
			else {
				i_out.write(ibuf[i-16]);
				q_out.write(qbuf[i-16]);
			}
//			printf("i_fft2_out[%d] = %f,\t q_fft2_out[%d] = %f\n", i, (float)ibuf[i], i, (float)qbuf[i]);
		}
	}

/*// Step 4.3 (Diamond-shaped Alpha Profile)
void alpha_profile(
	stream<single_fixed_t4 >& i_in,
	stream<single_fixed_t4 >& q_in,
	stream<single_fixed_t5 >& alpha_out,
	const unsigned R // 256
) {
// #pragma HLS DATAFLOW
	static single_fixed_t5 alpha_mem[512][8]; // size: 512*8 = 4096
//	single_fixed_t5 alpha_mem[512][8]; // size: 512*8 = 4096
// #pragma HLS ARRAY_PARTITION variable=alpha_mem complete dim=2

alpha_profile:	for (int k = 0; k < R; k++) { // Diamond downward direction (256)
		for (int r = 0; r < R; r++) { // Diamond upward direction (256)
#pragma HLS PIPELINE II=1
			for (int i = 0; i < 32; i++) {
#pragma HLS UNROLL
				single_fixed_t4 i_tmp = i_in.read();
				single_fixed_t4 q_tmp = q_in.read();
				single_fixed_t5 alpha_tmp[16];

				** Use middle range [8, 23] of the 32-point FFT outputs **
				if (i >= 8 && i < 24) {
					// sqr_tmp = i_in.read()*i_in.read() + q_in.read()*q_in.read();
					double_fixed_t4 ii_tmp = i_tmp * i_tmp;
					double_fixed_t4 qq_tmp = q_tmp * q_tmp;
					double_fixed_t4 sqr_tmp = ii_tmp + qq_tmp;
//					alpha_tmp[23-i] = sqr_tmp; // fxp_sqrt_top(sqr_tmp);
					// Xilinx solution -> https://www.xilinx.com/support/answers/72340.html
					float temp;
					temp = (float) sqr_tmp;
					alpha_tmp[23-i] = hls::sqrt(temp);
					if (i >= 16) { // Pb: [23, 16]
						if (alpha_tmp[23-i] > alpha_mem[R-1-r+k][23-i]) {
							alpha_mem[R-1-r+k][23-i] = alpha_tmp[23-i];
						}
					}
					if (i < 16) { // Pa: [15, 8]
						if (alpha_tmp[23-i] > alpha_mem[R-r+k][15-i]) {
							alpha_mem[R-r+k][15-i] = alpha_tmp[23-i];
						}
					}
					if (i < 16) { // Pb: [8, 15]
						if (alpha_tmp[i-8] > alpha_mem[R-1-r+k][i-8]) {
							alpha_mem[R-1-r+k][i-8] = alpha_tmp[i-8];
						}
					}
					if (i >= 16) { // Pa: [16, 23]
						if (alpha_tmp[i-8] > alpha_mem[R-r+k][i-16]) {
							alpha_mem[R-r+k][i-16] = alpha_tmp[i-8];
						}
					}
//					if (k == 5 && 0 < r < 10) {
//						printf("i_tmp[%d][%d] = %f, q_tmp[%d][%d] = %f\n", i, r, (float)i_tmp, i, r, (float)q_tmp);
//						printf("sqr_tmp[%d][%d] = %f\n", (R-1-r+k), (i-8), (float)sqr_tmp);
//						printf("alpha_tmp[%d][%d] = %f\n", i-8, r, (float)alpha_tmp[i-8]);
//						printf("alpha_mem[%d][%d] = %f\n", R-1-r+k, 23-i, (float)alpha_mem[R-1-r+k][23-i]);
//					}
				}

				** Use top/bottom range [0,7] & [24,31] of the 32-point FFT outputs **
				if (i>=0 && i<8)
					alpha_mem[r][16-i] = i_in.read()*i_in.read() + q_in.read()*q_in.read();
				if (i>=24 && i<32)
					alpha_mem[r][i-16] = i_in.read()*i_in.read() + q_in.read()*q_in.read();

			}
		}
	}
alpha_out:	for (int i = 0; i < 512; i++)
		for (int j = 0; j < 8; j++)
#pragma HLS UNROLL
			alpha_out.write(alpha_mem[i][j]);
}*/

void scd_matrix(
	stream<single_fixed_t2 >& i_in,
	stream<single_fixed_t2 >& q_in,
    stream<single_fixed_t4 >& i_out,
	stream<single_fixed_t4 >& q_out
){
#pragma HLS DATAFLOW

	stream<single_fixed_t3 > i_conjugate_out("i_conjugate_out_stream");
	stream<single_fixed_t3 > q_conjugate_out("q_conjugate_out_stream");
	stream<single_fixed_t3 > i_rev_32_out("i_rev_32_out_stream");
	stream<single_fixed_t3 > q_rev_32_out("q_rev_32_out_stream");

    conjugate_mult(i_in, q_in, i_conjugate_out, q_conjugate_out); // Step 4.1
//    fft_ip_32(i_conjugate_out, q_conjugate_out, i_out, q_out);
    bit_reverser_32(i_conjugate_out, q_conjugate_out, i_rev_32_out, q_rev_32_out); // Step 4.2.1
    fft_32(i_rev_32_out, q_rev_32_out, i_out, q_out); // Step 4.2.2
//    alpha_profile(i_fft_32_out, q_fft_32_out, alpha_out, 256); // Step 4.3

}

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

	const int in_size = 32;
	const int out_size = 32;
	stream<single_fixed_t2 > i_in;
	stream<single_fixed_t2 > q_in;
	stream<single_fixed_t4 > i_out;
	stream<single_fixed_t4 > q_out;

INPUT:	for (int i = 0; i < in_size; i++) {
#pragma HLS PIPELINE II=1
		fconvert i_conv, q_conv;
		i_conv.f = axis_i_in.read()/10.0; // /10.0
		q_conv.f = axis_q_in.read()/10.0; // /10.0
		i_in.write(i_conv.f);
		q_in.write(q_conv.f);
	}

	scd_matrix(i_in, q_in, i_out, q_out);

OUTPUT:	for (int i = 0; i < out_size; i++) {
#pragma HLS PIPELINE II=1
		fconvert i_conv, q_conv;
		axis_out i_tmp_out, q_tmp_out;
		i_conv.f = i_out.read();
		q_conv.f = q_out.read();
		i_tmp_out.data = i_conv.i;
		q_tmp_out.data = q_conv.i;

//		printf("i__out_float[%d] = %f, i_int[%d] = %d, i_hex[%d] = %x\n", i, i_conv.f, i, i_conv.i, i, i_tmp_out.data);

 		if(i == (out_size-1)) {
 			i_tmp_out.last = 1;
 			q_tmp_out.last = 1;
 		}
		else {
			i_tmp_out.last = 0;
			q_tmp_out.last = 0;
		}
		axis_i_out.write(i_tmp_out);
		axis_q_out.write(q_tmp_out);
	}
}
