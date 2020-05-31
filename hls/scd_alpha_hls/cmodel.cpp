#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"
#include "hls_stream.h"

#include "exp_table.h"
#include "window.h"
#include "types.h"

#include "fft_top.h"
#include "fxp_sqrt_top.h"

using namespace hls;

/*** Define AXI-Stream structures ***/
struct axis_in{
	int data; // single_fixed_t data
	bool last;
};
struct axis_out{
	unsigned int data; // single_fixed_t data
	bool last;
};
typedef hls::stream<axis_in> AXIS_IN;
typedef hls::stream<axis_out> AXIS_OUT;

typedef union {
	unsigned int i;
	float f; // float f;
} fconvert;

/********** Preprocess **********/
// Step 1.1 (Framing)
void framing(
	stream<single_fixed_t >& i_in,
	stream<single_fixed_t >& q_in,
	stream<single_fixed_t >& i_out,
	stream<single_fixed_t >& q_out,
	const unsigned P // 32
) {
	int Np = 256; // 256
	int L = Np/4; // 64
	single_fixed_t i_shift_reg[256];
	single_fixed_t q_shift_reg[256];
#pragma HLS ARRAY_PARTITION variable=i_shift_reg complete dim=0
#pragma HLS ARRAY_PARTITION variable=q_shift_reg complete dim=0

Shift_Reg_Loop: for (int i = 0; i <= (P-1)*L+Np; i++) { // (32-1)*64 + 256 = 2240
#pragma HLS PIPELINE II=1
		for (int j = Np-1; j >= 0; j--) {
// #pragma HLS UNROLL
			if (i >= Np && (i-Np)%L == 0) {
				i_out.write(i_shift_reg[j]); // i_shift_reg[Np-1-j]
				q_out.write(q_shift_reg[j]); // q_shift_reg[Np-1-j]
			}
			if (j == 0) {
				i_shift_reg[0] = i_in.read();
				q_shift_reg[0] = q_in.read();
			} else {
				i_shift_reg[j] = i_shift_reg[j-1];
				q_shift_reg[j] = q_shift_reg[j-1];
			}
		}
	}
}

// Step 1.2 (Hamming windowing)
void windowing(
    stream<single_fixed_t >& i_in,
    stream<single_fixed_t >& q_in,
	stream<single_fixed_t >& i_out,
	stream<single_fixed_t >& q_out,
	const unsigned P // 32
) {
	for (int p = 0; p < P; p++) {
#pragma HLS PIPELINE II=1
		int Np = 256; // 256
		single_fixed_t ibuf[256];
		single_fixed_t qbuf[256];

		for (int i = 0; i < Np; i++) {
#pragma HLS UNROLL
			// multiply with hamming-window
			single_fixed_t i_temp = i_in.read();
			single_fixed_t q_temp = q_in.read();
			ibuf[i] = i_temp * hamming[i];
			qbuf[i] = q_temp * hamming[i];
		}

		for (int i = 0; i < Np; i++) {
#pragma HLS UNROLL
			i_out.write(ibuf[i]);
			q_out.write(qbuf[i]);
//			printf("i_window[%d] = %f,\t q_window[%d] = %f\n", i, (float)ibuf[i], i, (float)qbuf[i]);
		}
	}
}

// Step 2 (256-pt FFT using Xilinx IP core)
void fft_ip_256(
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
	fft_config.setDir(1); // Forward(1); Inverse(0)
	// Set FFT length to 256 => log2(256) = 8
	fft_config.setSch(0xAA); // [2 2 2 2]
#pragma HLS interface ap_fifo depth=256 port=xn
#pragma HLS interface ap_fifo depth=256 port=xk
#pragma HLS interface ap_fifo port=fft_config
// #pragma HLS interface ap_fifo port=fft_status
#pragma HLS data_pack variable=xn
#pragma HLS data_pack variable=xk
#pragma HLS data_pack variable=fft_config

	for (int p = 0; p < P; p++) {
#pragma HLS PIPELINE II=1
		int Np = 256;
		for (int i = 0; i < Np; i++) {
#pragma HLS UNROLL
			data_in_t re, im;
			re = i_in.read();
			im = q_in.read();
			xn[i] = cmpxDataIn(re, im);
		}
//		dummy_proc_fe(1, &fft_config);
		// FFT IP
		hls::fft<config1>(xn, xk, &fft_status, &fft_config);
		// FFT-shift (shifting the zero-frequency component to the center of the array)
		for (int i = 0; i < Np; i++) {
#pragma HLS UNROLL
//			data_out_t re, im;
			cmpxDataOut out_temp[FFT_LENGTH];
			out_temp[i] = xk[i];
//			if (i < 128) {
//				i_out.write(out_temp[i+128].real() << 3);
//				q_out.write(out_temp[i+128].imag() << 3);
//			}
//			else {
//				i_out.write(out_temp[i-128].real() << 3);
//				q_out.write(out_temp[i-128].imag() << 3);
//			}
			i_out.write(out_temp[i].real() ); // << 3
			q_out.write(out_temp[i].imag() ); // << 3
//			printf("i_fft_out[%d] = %f,\t q_fft_out[%d] = %f\n", i, (float)xk[i].real(), i, (float)xk[i].imag());
		}
	}
}


// Step 2.1 (Bit reverse for 256-pt FFT)
void bit_reverser_256(
    stream<single_fixed_t >& i_in,
    stream<single_fixed_t >& q_in,
	stream<single_fixed_t >& i_out,
	stream<single_fixed_t >& q_out,
	const unsigned P // 32
) {
	const unsigned indices[256] = {
	0,128,64,192,32,160,96,224,16,144,80,208,48,176,112,240,
	8,136,72,200,40,168,104,232,24,152,88,216,56,184,120,248,
	4,132,68,196,36,164,100,228,20,148,84,212,52,180,116,244,
	12,140,76,204,44,172,108,236,28,156,92,220,60,188,124,252,
	2,130,66,194,34,162,98,226,18,146,82,210,50,178,114,242,
	10,138,74,202,42,170,106,234,26,154,90,218,58,186,122,250,
	6,134,70,198,38,166,102,230,22,150,86,214,54,182,118,246,
	14,142,78,206,46,174,110,238,30,158,94,222,62,190,126,254,
	1,129,65,193,33,161,97,225,17,145,81,209,49,177,113,241,
	9,137,73,201,41,169,105,233,25,153,89,217,57,185,121,249,
	5,133,69,197,37,165,101,229,21,149,85,213,53,181,117,245,
	13,141,77,205,45,173,109,237,29,157,93,221,61,189,125,253,
	3,131,67,195,35,163,99,227,19,147,83,211,51,179,115,243,
	11,139,75,203,43,171,107,235,27,155,91,219,59,187,123,251,
	7,135,71,199,39,167,103,231,23,151,87,215,55,183,119,247,
	15,143,79,207,47,175,111,239,31,159,95,223,63,191,127,255
	};

	for (int p = 0; p < P; p++) {
#pragma HLS PIPELINE II=1

		int Np = 256;
		single_fixed_t ibuf[256];
		single_fixed_t qbuf[256];

		for (int i = 0; i < Np; i++) {
#pragma HLS UNROLL
			ibuf[i] = i_in.read();
			qbuf[i] = q_in.read();
		}

		for (int i = 0; i < Np; i++) {
#pragma HLS UNROLL
			i_out.write(ibuf[indices[i]]);
			q_out.write(qbuf[indices[i]]);
//			if (p == 0) {
//				printf("i_rev[%d] = %f,\t q_rev[%d] = %f\n", i, (float)ibuf[indices[i]], i, (float)qbuf[indices[i]]);
//			}
		}
	}
}

// Step 2.2 (256-point FFT in the time domain & FFT-shift)
void fft_256(
	stream<single_fixed_t >& i_in,
	stream<single_fixed_t >& q_in,
	stream<single_fixed_t1 >& i_out,
	stream<single_fixed_t1 >& q_out,
	const unsigned num_ffts // 32
) {
FFT_256: for (int r = 0; r < num_ffts; r++) {
//#pragma HLS PIPELINE II=1

		double_fixed_t ibuf[256];
		double_fixed_t qbuf[256];

		for (int i = 0; i < 256; i++) {
#pragma HLS UNROLL
			ibuf[i] = i_in.read();
			qbuf[i] = q_in.read();
//			if (r == 0) {
//				printf("i_fft_in[%d] = %f,\t q_fft_in[%d] = %f\n", i, (float)ibuf[i], i, (float)qbuf[i]);
//			}
		}

		// size = 2
		int halfsize = 1; // halfsize of butterfly
		int tablestep = 128; // 32
		for (int i = 0; i < 256; i += 2) {
// #pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
// #pragma HLS PIPELINE II=1
				double_fixed_t temp_ii = single_fixed_t1 (ibuf[j+halfsize]) * exp_table_256_i[k];
				double_fixed_t temp_qq = single_fixed_t1 (qbuf[j+halfsize]) * exp_table_256_q[k];
				double_fixed_t temp_i = temp_ii - temp_qq;
				double_fixed_t temp_iq = single_fixed_t1 (ibuf[j+halfsize]) * exp_table_256_q[k];
				double_fixed_t temp_qi = single_fixed_t1 (qbuf[j+halfsize]) * exp_table_256_i[k];
				double_fixed_t temp_q = temp_iq + temp_qi;

//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_256_i[k]) - (qbuf[j+halfsize] * exp_table_256_q[k])) ;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_256_q[k]) + (qbuf[j+halfsize] * exp_table_256_i[k])) ;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) >> 1; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) >> 1; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) >> 1; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) >> 1; // >> 1
				k += tablestep;
			}
		}

		// size = 4
		halfsize = 2;
		tablestep = 64; // 16
		for (int i = 0; i < 256; i += 4) {
// #pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
// #pragma HLS PIPELINE II=1
				double_fixed_t temp_ii = single_fixed_t1 (ibuf[j+halfsize]) * exp_table_256_i[k];
				double_fixed_t temp_qq = single_fixed_t1 (qbuf[j+halfsize]) * exp_table_256_q[k];
				double_fixed_t temp_i = temp_ii - temp_qq;
				double_fixed_t temp_iq = single_fixed_t1 (ibuf[j+halfsize]) * exp_table_256_q[k];
				double_fixed_t temp_qi = single_fixed_t1 (qbuf[j+halfsize]) * exp_table_256_i[k];
				double_fixed_t temp_q = temp_iq + temp_qi;

//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_256_i[k]) - (qbuf[j+halfsize] * exp_table_256_q[k])) ;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_256_q[k]) + (qbuf[j+halfsize] * exp_table_256_i[k])) ;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) >> 1; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) >> 1; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) >> 1; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) >> 1; // >> 1
				k += tablestep;
			}
		}

		// size = 8
		halfsize = 4;
		tablestep = 32; // 8
		for (int i = 0; i < 256; i += 8) {
// #pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
// #pragma HLS PIPELINE II=1
				double_fixed_t temp_ii = single_fixed_t1 (ibuf[j+halfsize]) * exp_table_256_i[k];
				double_fixed_t temp_qq = single_fixed_t1 (qbuf[j+halfsize]) * exp_table_256_q[k];
				double_fixed_t temp_i = temp_ii - temp_qq;
				double_fixed_t temp_iq = single_fixed_t1 (ibuf[j+halfsize]) * exp_table_256_q[k];
				double_fixed_t temp_qi = single_fixed_t1 (qbuf[j+halfsize]) * exp_table_256_i[k];
				double_fixed_t temp_q = temp_iq + temp_qi;

//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_256_i[k]) - (qbuf[j+halfsize] * exp_table_256_q[k])) ;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_256_q[k]) + (qbuf[j+halfsize] * exp_table_256_i[k])) ;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) >> 1; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) >> 1; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) >> 1; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) >> 1; // >> 1
				k += tablestep;
			}
		}

		// size = 16
		halfsize = 8;
		tablestep = 16; // 4
		for (int i = 0; i < 256; i += 16) {
// #pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
// #pragma HLS PIPELINE II=1
				double_fixed_t temp_ii = single_fixed_t1 (ibuf[j+halfsize]) * exp_table_256_i[k];
				double_fixed_t temp_qq = single_fixed_t1 (qbuf[j+halfsize]) * exp_table_256_q[k];
				double_fixed_t temp_i = temp_ii - temp_qq;
				double_fixed_t temp_iq = single_fixed_t1 (ibuf[j+halfsize]) * exp_table_256_q[k];
				double_fixed_t temp_qi = single_fixed_t1 (qbuf[j+halfsize]) * exp_table_256_i[k];
				double_fixed_t temp_q = temp_iq + temp_qi;

//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_256_i[k]) - (qbuf[j+halfsize] * exp_table_256_q[k])) ;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_256_q[k]) + (qbuf[j+halfsize] * exp_table_256_i[k])) ;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) >> 1; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) >> 1; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) >> 1; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) >> 1; // >> 1
				k += tablestep;
			}
		}

		// size = 32
		halfsize = 16;
		tablestep = 8; // 2
		for (int i = 0; i < 256; i += 32) {
// #pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
// #pragma HLS PIPELINE II=1
				double_fixed_t temp_ii = single_fixed_t1 (ibuf[j+halfsize]) * exp_table_256_i[k];
				double_fixed_t temp_qq = single_fixed_t1 (qbuf[j+halfsize]) * exp_table_256_q[k];
				double_fixed_t temp_i = temp_ii - temp_qq;
				double_fixed_t temp_iq = single_fixed_t1 (ibuf[j+halfsize]) * exp_table_256_q[k];
				double_fixed_t temp_qi = single_fixed_t1 (qbuf[j+halfsize]) * exp_table_256_i[k];
				double_fixed_t temp_q = temp_iq + temp_qi;

//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_256_i[k]) - (qbuf[j+halfsize] * exp_table_256_q[k])) ;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_256_q[k]) + (qbuf[j+halfsize] * exp_table_256_i[k])) ;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) >> 1; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) >> 1; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) >> 1; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) >> 1; // >> 1
				k += tablestep;
			}
		}

		// size = 64
		halfsize = 32;
		tablestep = 4; // 1
		for (int i = 0; i < 256; i += 64) {
// #pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
// #pragma HLS PIPELINE II=1
				double_fixed_t temp_ii = single_fixed_t1 (ibuf[j+halfsize]) * exp_table_256_i[k];
				double_fixed_t temp_qq = single_fixed_t1 (qbuf[j+halfsize]) * exp_table_256_q[k];
				double_fixed_t temp_i = temp_ii - temp_qq;
				double_fixed_t temp_iq = single_fixed_t1 (ibuf[j+halfsize]) * exp_table_256_q[k];
				double_fixed_t temp_qi = single_fixed_t1 (qbuf[j+halfsize]) * exp_table_256_i[k];
				double_fixed_t temp_q = temp_iq + temp_qi;

//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_256_i[k]) - (qbuf[j+halfsize] * exp_table_256_q[k])) ;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_256_q[k]) + (qbuf[j+halfsize] * exp_table_256_i[k])) ;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) >> 1; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) >> 1; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) >> 1; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) >> 1; // >> 1
				k += tablestep;
			}
		}

		// size = 128
		halfsize = 64;
		tablestep = 2;
		for (int i = 0; i < 256; i += 128) {
// #pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
// #pragma HLS PIPELINE II=1
				double_fixed_t temp_ii = single_fixed_t1 (ibuf[j+halfsize]) * exp_table_256_i[k];
				double_fixed_t temp_qq = single_fixed_t1 (qbuf[j+halfsize]) * exp_table_256_q[k];
				double_fixed_t temp_i = temp_ii - temp_qq;
				double_fixed_t temp_iq = single_fixed_t1 (ibuf[j+halfsize]) * exp_table_256_q[k];
				double_fixed_t temp_qi = single_fixed_t1 (qbuf[j+halfsize]) * exp_table_256_i[k];
				double_fixed_t temp_q = temp_iq + temp_qi;

//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_256_i[k]) - (qbuf[j+halfsize] * exp_table_256_q[k])) ;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_256_q[k]) + (qbuf[j+halfsize] * exp_table_256_i[k])) ;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) >> 1; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) >> 1; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) >> 1; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) >> 1; // >> 1
				k += tablestep;
			}
		}

		// size = 256
		halfsize = 128;
		tablestep = 1;
		for (int i = 0; i < 256; i += 256) {
// #pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
// #pragma HLS PIPELINE II=1
				double_fixed_t temp_ii = single_fixed_t1 (ibuf[j+halfsize]) * exp_table_256_i[k];
				double_fixed_t temp_qq = single_fixed_t1 (qbuf[j+halfsize]) * exp_table_256_q[k];
				double_fixed_t temp_i = temp_ii - temp_qq;
				double_fixed_t temp_iq = single_fixed_t1 (ibuf[j+halfsize]) * exp_table_256_q[k];
				double_fixed_t temp_qi = single_fixed_t1 (qbuf[j+halfsize]) * exp_table_256_i[k];
				double_fixed_t temp_q = temp_iq + temp_qi;

//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_256_i[k]) - (qbuf[j+halfsize] * exp_table_256_q[k])) ;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_256_q[k]) + (qbuf[j+halfsize] * exp_table_256_i[k])) ;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) >> 1; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) >> 1; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) >> 1; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) >> 1; // >> 1
				k += tablestep;
			}
		}

		for (int i = 0; i < 256; i++) {
#pragma HLS UNROLL
			// Direct write out the 256-point FFT outputs
			i_out.write(ibuf[i]);
			q_out.write(qbuf[i]);
			// FFT-shift (shifting the zero-frequency component to the center of the array)
//			if (i < 128) {
//				i_out.write(ibuf[i+128]);
//				q_out.write(qbuf[i+128]);
//			}
//			else {
//				i_out.write(ibuf[i-128]);
//				q_out.write(qbuf[i-128]);
//			}
//			if (r == 0) {
//				printf("i_fft1_out[%d] = %f,\t q_fft1_out[%d] = %f\n", i, (float)ibuf[i], i, (float)qbuf[i]);
//			}
		}
	}
}

// Step 3 (Down Conversion & FFT Shift)
void down_conversion(
	stream<single_fixed_t1 >& i_in,
	stream<single_fixed_t1 >& q_in,
	stream<single_fixed_t2 >& i_out,
	stream<single_fixed_t2 >& q_out,
	const unsigned P // 32
) {
	single_fixed_t2 i_buf[256];
	single_fixed_t2 q_buf[256];
	int Np = 256;
	for (int k = 0; k < P; k++) { // frames [0, P-1]
#pragma HLS PIPELINE II=1
		for (int m = 0; m < Np; m++) { // frequencies [-Np/2, Np/2-1]
#pragma HLS UNROLL
			single_fixed_t1 i_temp = i_in.read();
			single_fixed_t1 q_temp = q_in.read();
			// out = out * exp(-i*2*PI*k*m*L/Np) = out * exp(-i*PI*k*m/2)
			int phase = k*m; // k*(m-128)
			if (phase%4 == 0) { // 2*k*PI: cos = 1, sin = 0.
				i_buf[m] = i_temp;
				q_buf[m] = 0;
			}
			if (phase%4 == 1) { // (2*k+1)/2*PI: cos = 0, sin = 1.
				i_buf[m] = 0;
				q_buf[m] = q_temp;
			}
			if (phase%4 == 2) { // (2*k+1)/2*PI: cos = 0, sin = 1.
				i_buf[m] = -i_temp;
				q_buf[m] = 0;
			}
			if (phase%4 == 3) { // (2*k+1)/2*PI: cos = 0, sin = 1.
				i_buf[m] = 0;
				q_buf[m] = -q_temp;
			}
//			if (k == 1) {
//				printf("theta = %f,\t cos[%d][%d] = %f,\t sin[%d][%d] = %f\n", (float)theta, k, m, (float)cos_term, k, m, (float)sin_term);
//			}
		}
		for (int j = 0; j < Np; j++) {
#pragma HLS UNROLL
//			i_out.write(i_buf[j]);
//			q_out.write(q_buf[j]);
			// FFT-shift (shifting the zero-frequency component to the center of the array)
			if (j < 128) {
				i_out.write(i_buf[j+128] );
				q_out.write(q_buf[j+128] );
			}
			else {
				i_out.write(i_buf[j-128] );
				q_out.write(q_buf[j-128] );
			}
//			printf("i_dc[%d] = %f,\t q_dc[%d] = %f\n", j, (float)i_buf[j], j, (float)q_buf[j]);
		}
	}
}

/*
// Step 3 (Down Conversion)
void down_conversion(
	stream<single_fixed_t1 >& i_in,
	stream<single_fixed_t1 >& q_in,
	stream<single_fixed_t2 >& i_out,
	stream<single_fixed_t2 >& q_out,
	const unsigned P // 32
) {
	double_fixed_t i_buf[256];
	double_fixed_t q_buf[256];
	int Np = 256;
	for (int k = 0; k < P; k++) { // frames [0, P-1]
#pragma HLS PIPELINE II=1
		for (int m = 0; m < Np; m++) { // frequencies [-Np/2, Np/2-1]
#pragma HLS UNROLL
			single_fixed_t1 i_temp = i_in.read();
			single_fixed_t1 q_temp = q_in.read();
			// out = out * exp(-i*2*pi*k*m*L/Np) = out * exp(-i*pi*k*m/2)
			float theta = float (k*(m-128)/2.0); // (k*(m-128)/2.0)
			exp_fixed_t cos_term = hls::cospi(theta);
			exp_fixed_t sin_term = hls::sinpi(theta);
			double_fixed_t1 temp_ii = single_fixed_t1 (i_temp) * cos_term;
			double_fixed_t1 temp_qq = single_fixed_t1 (q_temp) * sin_term;
			double_fixed_t1 temp_iq = single_fixed_t1 (i_temp) * sin_term;
			double_fixed_t1 temp_qi = single_fixed_t1 (q_temp) * cos_term;
			i_buf[m] = temp_ii + temp_qq; // I*cos(theta) + Q*sin(theta)
			q_buf[m] = temp_qi - temp_iq; // Q*cos(theta) - I*sin(theta)
//			i_buf[m] = single_fixed_t (float (i_temp) * cospi(float (k*(m-128)/2)) + float (q_temp) * sinpi(float (k*(m-128)/2)));
//			q_buf[m] = single_fixed_t (float (q_temp) * cospi(float (k*(m-128)/2)) - float (i_temp) * sinpi(float (k*(m-128)/2)));
//			if (k == 1) {
//				printf("theta = %f,\t cos[%d][%d] = %f,\t sin[%d][%d] = %f\n", (float)theta, k, m, (float)cos_term, k, m, (float)sin_term);
//			}
		}
		for (int j = 0; j < Np; j++) {
#pragma HLS UNROLL
			i_out.write(i_buf[j]);
			q_out.write(q_buf[j]);
//			printf("i_dc[%d] = %f,\t q_dc[%d] = %f\n", j, (float)i_buf[j], j, (float)q_buf[j]);
		}
	}
}
*/

/********** SCD Matrix **********/
// Step 4.1
void conjugate_mult(
	stream<single_fixed_t2 >& i_in,
	stream<single_fixed_t2 >& q_in,
	stream<single_fixed_t3 >& i_out,
	stream<single_fixed_t3 >& q_out,
	const unsigned Np // 256
) {
#pragma HLS DATAFLOW
/*** Two Memories for I/Q stream ***/
	single_fixed_t2 i_mem[256][32];
	single_fixed_t2 q_mem[256][32];
//	single_fixed_t1 i_mem[256][32];
//	single_fixed_t1 q_mem[256][32];
	int P = 32;
	for (int i = 0; i < P; i++) {
#pragma HLS PIPELINE II=1
		for (int j = 0; j < Np; j++) {
#pragma HLS UNROLL
			i_mem[j][i] = i_in.read();
			q_mem[j][i] = q_in.read();
		}
	}
	/*** Y <- X[i] .* conjugate(X[j]) ***/
Mult:	for (int k = 0; k < Np; k++){ // original (X) row loop
		for (int i = 0; i < Np; i++) { // conjugate (X*) row loop
#pragma HLS PIPELINE II=1
			for (int j = 0; j < P; j++) { // column loop (both)
#pragma HLS UNROLL
				double_fixed_t2 ii_tmp = i_mem[k][j] * i_mem[i][j];
				double_fixed_t2 qq_tmp = q_mem[k][j] * q_mem[i][j];
				double_fixed_t2 iq_tmp = i_mem[k][j] * q_mem[i][j];
				double_fixed_t2 qi_tmp = q_mem[k][j] * i_mem[i][j];
				double_fixed_t2 i_tmp = (ii_tmp + qq_tmp);
				double_fixed_t2 q_tmp = (qi_tmp - iq_tmp);

//				single_fixed_t i_tmp = i_mem[k][j] * i_mem[i][j] + q_mem[k][j] * q_mem[i][j];
//				single_fixed_t q_tmp = q_mem[k][j] * i_mem[i][j] - i_mem[k][j] * q_mem[i][j];
				i_out.write(i_tmp);
				q_out.write(q_tmp);
			}
		}
	}
}

// Step 4.2.1 (Bit reversal for 32-point FFT)
void bit_reverser_32(
    stream<single_fixed_t3 >& i_in,
    stream<single_fixed_t3 >& q_in,
    stream<single_fixed_t3 >& i_out,
    stream<single_fixed_t3 >& q_out,
	const unsigned num_rows // 65536
) {
	const unsigned indices[32] = {
	0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,
	1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31
	};

	for (int r = 0; r < num_rows; r++) {
#pragma HLS PIPELINE II=1
		int P = 32;
		single_fixed_t3 ibuf[32];
		single_fixed_t3 qbuf[32];
		for (int i = 0; i < P; i++) {
#pragma HLS UNROLL
			ibuf[i] = i_in.read();
			qbuf[i] = q_in.read();
		}
		for (int i = 0; i < P; i++) {
#pragma HLS UNROLL
			i_out.write(ibuf[indices[i]]);
			q_out.write(qbuf[indices[i]]);
		}
	}
}

// Step 4.2.2 (32-point FFT in time-lag & FFT-shift)
void fft_32(
    stream<single_fixed_t3 >& i_in,
    stream<single_fixed_t3 >& q_in,
    stream<single_fixed_t4 >& i_out,
	stream<single_fixed_t4 >& q_out,
	const unsigned num_ffts // 256*256 = 65536
) {
FFT_32: for (int r = 0; r < num_ffts; r++) {
#pragma HLS PIPELINE II=1
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
#pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
#pragma HLS PIPELINE II=1
				double_fixed_t3 temp_ii = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_qq = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_i = temp_ii - temp_qq;
				double_fixed_t3 temp_iq = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_qi = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_q = temp_iq + temp_qi;

//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_32_i[k]) - (qbuf[j+halfsize] * exp_table_32_q[k])) >> 14;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_32_q[k]) + (qbuf[j+halfsize] * exp_table_32_i[k])) >> 14;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) ; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) ; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) ; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) ; // >> 1
				k += tablestep;
			}
		}

		// size = 4
		halfsize = 2;
		tablestep = 8;

		for (int i = 0; i < 32; i += 4) {
#pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
#pragma HLS PIPELINE II=1
				double_fixed_t3 temp_ii = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_qq = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_i = temp_ii - temp_qq;
				double_fixed_t3 temp_iq = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_qi = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_q = temp_iq + temp_qi;

//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_32_i[k]) - (qbuf[j+halfsize] * exp_table_32_q[k])) >> 14;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_32_q[k]) + (qbuf[j+halfsize] * exp_table_32_i[k])) >> 14;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) ; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) ; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) ; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) ; // >> 1
				k += tablestep;
			}
		}

		// size = 8
		halfsize = 4;
		tablestep = 4;

		for (int i = 0; i < 32; i += 8) {
#pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
#pragma HLS PIPELINE II=1
				double_fixed_t3 temp_ii = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_qq = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_i = temp_ii - temp_qq;
				double_fixed_t3 temp_iq = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_qi = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_q = temp_iq + temp_qi;

//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_32_i[k]) - (qbuf[j+halfsize] * exp_table_32_q[k])) >> 14;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_32_q[k]) + (qbuf[j+halfsize] * exp_table_32_i[k])) >> 14;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) ; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) ; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) ; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) ; // >> 1
				k += tablestep;
			}
		}

		// size = 16
		halfsize = 8;
		tablestep = 2;

		for (int i = 0; i < 32; i += 16) {
#pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
#pragma HLS PIPELINE II=1
				double_fixed_t3 temp_ii = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_qq = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_i = temp_ii - temp_qq;
				double_fixed_t3 temp_iq = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_qi = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_q = temp_iq + temp_qi;

//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_32_i[k]) - (qbuf[j+halfsize] * exp_table_32_q[k])) >> 14;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_32_q[k]) + (qbuf[j+halfsize] * exp_table_32_i[k])) >> 14;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) ; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) ; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) ; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) ; // >> 1
				k += tablestep;
			}
		}

		// size = 32
		halfsize = 16;
		tablestep = 1;

		for (int i = 0; i < 32; i += 32) {
#pragma HLS UNROLL
			int k = 0;
			for (int j = i; j < (i+halfsize); j++) {
#pragma HLS PIPELINE II=1
				double_fixed_t3 temp_ii = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_qq = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_i = temp_ii - temp_qq;
				double_fixed_t3 temp_iq = single_fixed_t4 (ibuf[j+halfsize]) * exp_table_32_q[k];
				double_fixed_t3 temp_qi = single_fixed_t4 (qbuf[j+halfsize]) * exp_table_32_i[k];
				double_fixed_t3 temp_q = temp_iq + temp_qi;

//				single_fixed_t temp_i = ((ibuf[j+halfsize] * exp_table_32_i[k]) - (qbuf[j+halfsize] * exp_table_32_q[k])) >> 14;
//				single_fixed_t temp_q = ((ibuf[j+halfsize] * exp_table_32_q[k]) + (qbuf[j+halfsize] * exp_table_32_i[k])) >> 14;
				ibuf[j+halfsize] = (ibuf[j] - temp_i) ; // >> 1
				qbuf[j+halfsize] = (qbuf[j] - temp_q) ; // >> 1
				ibuf[j] = (ibuf[j] + temp_i) ; // >> 1
				qbuf[j] = (qbuf[j] + temp_q) ; // >> 1
				k += tablestep;
			}
		}

		// write out the FFT output
/*		for (int i = 0; i < 32; i++) {
#pragma HLS UNROLL
			i_out.write(ibuf[i]);
			q_out.write(qbuf[i]);
		} */
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
}

/*** CORDIC-based sqrt function
void sqrt_top(const hls::sqrt_input<InputWidth, DataFormat>::in &x,
              hls::sqrt_output<OutputWidth, DataFormat>::out &sqrtX){

  // Call square root function
  hls::sqrt<DataFormat,InputWidth,OutputWidth,RoundMode>(x, sqrtX);

} ***/

out_data_t fxp_sqrt_top(in_data_t& in_val)
{
   out_data_t result;
   fxp_sqrt(result, in_val);
   return result;
}

// Step 4.3 (Diamond-shaped Alpha Profile)
void alpha_profile(
	stream<single_fixed_t4 >& i_in,
	stream<single_fixed_t4 >& q_in,
	stream<single_fixed_t5 >& alpha_out,
	const unsigned R // 256
) {
// #pragma HLS DATAFLOW
	static single_fixed_t5 alpha_mem[512][8]; // size: 512*8 = 4096
//	single_fixed_t5 alpha_mem[512][8]; // size: 512*8 = 4096
#pragma HLS ARRAY_PARTITION variable=alpha_mem complete dim=2

alpha_profile:	for (int k = 0; k < R; k++) { // Diamond downward direction (256)
		for (int r = 0; r < R; r++) { // Diamond upward direction (256)
#pragma HLS PIPELINE II=1
			for (int i = 0; i < 32; i++) {
// #pragma HLS UNROLL
				single_fixed_t4 i_tmp = i_in.read();
				single_fixed_t4 q_tmp = q_in.read();
				single_fixed_t5 alpha_tmp[16];

				/*** Use middle range [8, 23] of the 32-point FFT outputs ***/
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
//					alpha_tmp[23-i] = sqr_tmp;
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
/*					if (i < 16) { // Pb: [8, 15]
						if (alpha_tmp[i-8] > alpha_mem[R-1-r+k][i-8]) {
							alpha_mem[R-1-r+k][i-8] = alpha_tmp[i-8];
						}
					}
					if (i >= 16) { // Pa: [16, 23]
						if (alpha_tmp[i-8] > alpha_mem[R-r+k][i-16]) {
							alpha_mem[R-r+k][i-16] = alpha_tmp[i-8];
						}
					} */
//					if (k == 5 && 0 < r < 10) {
//						printf("i_tmp[%d][%d] = %f, q_tmp[%d][%d] = %f\n", i, r, (float)i_tmp, i, r, (float)q_tmp);
//						printf("sqr_tmp[%d][%d] = %f\n", (R-1-r+k), (i-8), (float)sqr_tmp);
//						printf("alpha_tmp[%d][%d] = %f\n", i-8, r, (float)alpha_tmp[i-8]);
//						printf("alpha_mem[%d][%d] = %f\n", R-1-r+k, 23-i, (float)alpha_mem[R-1-r+k][23-i]);
//					}
				}

				/*** Use top/bottom range [0,7] & [24,31] of the 32-point FFT outputs ***/
/*				if (i>=0 && i<8)
					alpha_mem[r][16-i] = i_in.read()*i_in.read() + q_in.read()*q_in.read();
				if (i>=24 && i<32)
					alpha_mem[r][i-16] = i_in.read()*i_in.read() + q_in.read()*q_in.read();
*/
			}
		}
	}
alpha_out:	for (int i = 0; i < 512; i++)
		for (int j = 0; j < 8; j++)
#pragma HLS UNROLL
			alpha_out.write(alpha_mem[i][j]);
}


void preprocess(
	stream<single_fixed_t >& i_in,
	stream<single_fixed_t >& q_in,
	stream<single_fixed_t2 >& i_out,
	stream<single_fixed_t2 >& q_out
) {
#pragma HLS DATAFLOW

	stream<single_fixed_t > i_frame_out("i_frame_out_stream");
	stream<single_fixed_t > q_frame_out("q_frame_out_stream");

	stream<single_fixed_t > i_window_out("i_window_out_stream");
	stream<single_fixed_t > q_window_out("q_window_out_stream");

	stream<single_fixed_t > i_rev_256_out("i_rev_256_out_stream");
	stream<single_fixed_t > q_rev_256_out("q_rev_256_out_stream");

	stream<single_fixed_t1 > i_fft_256_out("i_fft_256_out_stream");
	stream<single_fixed_t1 > q_fft_256_out("q_fft_256_out_stream");

	framing(i_in, q_in, i_frame_out, q_frame_out, 32); // Step 1.1
	windowing(i_frame_out, q_frame_out, i_window_out, q_window_out, 32); // step 1.2
//	fft_ip_256(i_window_out, q_window_out, i_fft_256_out, q_fft_256_out, 32); //step 2
	bit_reverser_256(i_window_out, q_window_out, i_rev_256_out, q_rev_256_out, 32); // Step 2.1
	fft_256(i_rev_256_out, q_rev_256_out, i_fft_256_out, q_fft_256_out, 32); // Step 2.2
	down_conversion(i_fft_256_out, q_fft_256_out, i_out, q_out, 32); // Step 3

}


void scd_matrix(
	stream<single_fixed_t2 >& i_in,
	stream<single_fixed_t2 >& q_in,
	stream<single_fixed_t5 >& alpha_out
) {
#pragma HLS DATAFLOW

	stream<single_fixed_t3 > i_conjugate_out("i_conjugate_out_stream");
	stream<single_fixed_t3 > q_conjugate_out("q_conjugate_out_stream");

	stream<single_fixed_t3 > i_rev_32_out("i_rev_32_out_stream");
	stream<single_fixed_t3 > q_rev_32_out("q_rev_32_out_stream");

	stream<single_fixed_t4 > i_fft_32_out("i_fft_32_out_stream");
	stream<single_fixed_t4 > q_fft_32_out("q_fft_32_out_stream");

    conjugate_mult(i_in, q_in, i_conjugate_out, q_conjugate_out, 256); // Step 4.1
    bit_reverser_32(i_conjugate_out, q_conjugate_out, i_rev_32_out, q_rev_32_out, 65536); // Step 4.2.1
    fft_32(i_rev_32_out, q_rev_32_out, i_fft_32_out, q_fft_32_out, 65536); // Step 4.2.2
    alpha_profile(i_fft_32_out, q_fft_32_out, alpha_out, 256); // Step 4.3

}


void model(
	stream<single_fixed_t >& i_in,
	stream<single_fixed_t >& q_in,
	stream<single_fixed_t5 >& alpha_out
){
#pragma HLS DATAFLOW

	stream<single_fixed_t2 > i_preprocess_out("i_preprocess_out_stream");
	stream<single_fixed_t2 > q_preprocess_out("q_preprocess_out_stream");

	preprocess(i_in, q_in, i_preprocess_out, q_preprocess_out);
	scd_matrix(i_preprocess_out, q_preprocess_out, alpha_out);

}

// AXI Stream wrapper
void model_wrapper(
	stream<ap_int<16> >& axis_i_in,
	stream<ap_int<16> >& axis_q_in,
	AXIS_OUT &axis_alpha_out
){
// #pragma HLS INTERFACE s_axilite port=batch_size bundle=ctrl
// #pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=axis_i_in
#pragma HLS INTERFACE axis port=axis_q_in
#pragma HLS INTERFACE axis port=axis_alpha_out
#pragma HLS DATAFLOW

	const int in_size = 2240;
	const int out_size = 4096;
	stream<single_fixed_t > i_in;
	stream<single_fixed_t > q_in;
	stream<single_fixed_t2 > i_out;
	stream<single_fixed_t2 > q_out;
	stream<single_fixed_t2 > i_preprocess_out("i_preprocess_out_stream");
	stream<single_fixed_t2 > q_preprocess_out("q_preprocess_out_stream");
	stream<single_fixed_t5 > alpha_out;

	for (int j = 0; j <= in_size; j++) {
// #pragma HLS unroll factor=256
		// union {ap_int<16> i; single_fixed_t f;} i_tmp_in;
		// union {ap_int<16> i; single_fixed_t f;} q_tmp_in;
		// axis_in i_tmp_in, q_tmp_in;
		fconvert i_conv, q_conv;
		i_conv.f = axis_i_in.read()/10.0;
		q_conv.f = axis_q_in.read()/10.0;
		i_in.write(i_conv.f);
		q_in.write(q_conv.f);
		}

//	preprocess(i_in, q_in, i_preprocess_out, q_preprocess_out);
//	scd_matrix(i_preprocess_out, q_preprocess_out, alpha_out);

	model(i_in, q_in, alpha_out);

	for (int j = 0; j < out_size; j++) {
// #pragma HLS unroll factor=32
		fconvert alpha_conv;
		axis_out tmp_out;
		alpha_conv.f = alpha_out.read();
		tmp_out.data = alpha_conv.i;

		printf("alpha_float[%d] = %f, alpha_hex[%d] = %x\n", j, alpha_conv.f, j, tmp_out.data);

 		if(j == (out_size-1))
			tmp_out.last = 1;
		else
			tmp_out.last = 0;
		axis_alpha_out.write(tmp_out);
	}
}
