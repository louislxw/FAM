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

// Conjugate Multiplication
void conjugate_mult(
	stream<single_fixed_t2 >& i_in,
	stream<single_fixed_t2 >& q_in,
	stream<single_fixed_t3 >& i_out,
	stream<single_fixed_t3 >& q_out
){
#pragma HLS DATAFLOW

// Two registers for I/Q stream
	single_fixed_t2 i_mem[32];
	single_fixed_t2 q_mem[32];

	for (int i = 0; i < 32; i++) {
		i_mem[i] = i_in.read();
		q_mem[i] = q_in.read();
	}
// Complex multiplication requires 4 DSPs:
// i_tmp = i_mem[k][j] * i_mem[i][j] + q_mem[k][j] * q_mem[i][j];
// q_tmp = q_mem[k][j] * i_mem[i][j] - i_mem[k][j] * q_mem[i][j];
	for (int i = 0; i < 32; i++) {
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
void dummy_proc_fe(
	bool direction,
	config_t* config,
	cmpxData in[FFT_LENGTH],
	cmpxData out[FFT_LENGTH]
) {
	int i;
    config->setDir(direction); // 1: forward
    config->setSch(0x155); // pipelined_streaming_io: 0x1A = [1 10 10]; 0xAA = [10 10 10 10]
    for (i=0; i< FFT_LENGTH; i++)
    	out[i] = in[i];
}

void dummy_proc_be(
    status_t* status_in,
    bool* ovflo,
	cmpxData in[FFT_LENGTH],
	cmpxData out[FFT_LENGTH])
{
    int i;
    for (i=0; i< FFT_LENGTH; i++)
        out[i] = in[i];
    *ovflo = status_in->getOvflo() & 0x1;
}

void fft_top(
	bool direction,
	cmpxData in[FFT_LENGTH],
	cmpxData out[FFT_LENGTH],
	bool* ovflo
) {
//#pragma HLS interface ap_hs port=direction
//#pragma HLS interface ap_fifo depth=1 port=ovflo
#pragma HLS interface ap_fifo depth=FFT_LENGTH port=in,out
#pragma HLS data_pack variable=in
#pragma HLS data_pack variable=out
#pragma HLS DATAFLOW

	cmpxData xn[FFT_LENGTH];
	cmpxData xk[FFT_LENGTH];
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
	int P = 32;
	bool fft_direction = 1;
	bool fft_ovflo;
	cmpxData fft_in[FFT_LENGTH];
	cmpxData fft_out[FFT_LENGTH];
//#pragma HLS interface ap_hs port=fft_direction
//#pragma HLS interface ap_fifo depth=1 port=fft_ovflo
#pragma HLS interface ap_fifo depth=FFT_LENGTH port=fft_in,fft_out
#pragma HLS data_pack variable=fft_in
#pragma HLS data_pack variable=fft_out
#pragma HLS DATAFLOW

	for (int i = 0; i < P; i++) {
		single_fixed_t3 re, im;
		re = i_in.read();
		im = q_in.read();
		fft_in[i] = cmpxData(re, im);
	}

	fft_top(fft_direction, fft_in, fft_out, &fft_ovflo);

	for (int i = 0; i < P; i++) {
		cmpxData out_temp[FFT_LENGTH];
		out_temp[i] = fft_out[i];
		i_out.write(out_temp[i].real() );
		q_out.write(out_temp[i].imag() );
	}
}

void scd_pe(
	stream<single_fixed_t2 >& i_in,
	stream<single_fixed_t2 >& q_in,
    stream<single_fixed_t4 >& i_out,
	stream<single_fixed_t4 >& q_out
){
#pragma HLS DATAFLOW

	stream<single_fixed_t3 > i_conjugate_out("i_conjugate_out_stream");
	stream<single_fixed_t3 > q_conjugate_out("q_conjugate_out_stream");

    conjugate_mult(i_in, q_in, i_conjugate_out, q_conjugate_out); // 2-DSP or 4-DSP
    fft_ip_32(i_conjugate_out, q_conjugate_out, i_out, q_out); // 4-DSP
}

const int Np = 16;
const int P = 32;
const int PE = 16; // 256
void scd_pe_array(
	stream<single_fixed_t2 >& i_in,
	stream<single_fixed_t2 >& q_in,
    stream<single_fixed_t4 >& i_out,
	stream<single_fixed_t4 >& q_out
){
#pragma HLS DATAFLOW
	stream<single_fixed_t2 > i_pe_in[PE];
	stream<single_fixed_t2 > q_pe_in[PE];
	stream<single_fixed_t2 > i_pe_out[PE];
	stream<single_fixed_t2 > q_pe_out[PE];
#pragma HLS STREAM variable=i_pe_in depth=32 dim=1
#pragma HLS STREAM variable=q_pe_in depth=32 dim=1
#pragma HLS STREAM variable=i_pe_out depth=32 dim=1
#pragma HLS STREAM variable=q_pe_out depth=32 dim=1

	// Two registers for I/Q stream
	single_fixed_t2 i_mem[P][Np]; // 32*256
	single_fixed_t2 q_mem[P][Np]; // 32*256

	// I/Q stream feeds into the memories
//	for (int i = 0; i < Np; i++) {
//#pragma HLS PIPELINE II=1
//		for (int j = 0; j < P; j++) {
//			i_mem[j][i] = i_in.read();
//			q_mem[j][i] = q_in.read();
//	    }
//	}

	// outmost loop: (Np/PE)*(Np*PE) iterations
//	for (int ii = 0; ii < Np/PE; ii++) { // beginning of I stream iteration
//		for (int jj = 0; jj < Np/PE; jj++) { // beginning of Q stream iteration

			// Two registers for I/Q stream
			single_fixed_t2 i_shift_reg[P][PE]; // 32*PE
			single_fixed_t2 q_shift_reg[P][PE]; // 32*PE
#pragma HLS ARRAY_PARTITION variable=i_shift_reg complete dim=2
#pragma HLS ARRAY_PARTITION variable=q_shift_reg complete dim=2

			// 1. Initialize input data to shift registers
			for (int i = 0; i < PE; i++) {
#pragma HLS PIPELINE II=1
				for (int j = 0; j < P; j++) {
					i_shift_reg[j][i] = i_in.read(); // i_mem[j][i]; // +ii*PE
					q_shift_reg[j][i] = q_in.read(); // q_mem[j][i]; // +jj*PE
		    	}
			}
//#pragma HLS PIPELINE II=1
			// 2. PE array compute & shift register for PE times
			for (int k = 0; k < PE; k++) {
#pragma HLS PIPELINE II=1
				// 2.1 Computation on PE array
				for (int i = 0; i < PE; i++) {
					for (int j = 0; j < P; j++) {
						// pe_in buffers
						i_pe_in[i].write(i_shift_reg[j][i]);
						q_pe_in[i].write(q_shift_reg[j][i]);
					}
				    fft_ip_32(i_pe_in[i], q_pe_in[i], i_pe_out[i], q_pe_out[i]); // 4-DSP
//					scd_pe(i_pe_in, q_pe_in, i_out, q_out);
					for (int j = 0; j < P; j++) {
						// pe_out buffers
						single_fixed_t2 i_buf = i_pe_out[i].read();
						single_fixed_t2 q_buf = q_pe_out[i].read();
						i_out.write(i_buf);
						q_out.write(q_buf);
					}
				}
				// 2.2 Update data in shift registers
#pragma HLS PIPELINE II=1
Shift_Reg_Loop:	for (int i = PE-1; i >= 0; i--) {
//#pragma HLS PIPELINE II=1
					for (int j = 0; j < P; j++) {
//						if (i >= PE) {
//							i_pe_in.write(i_shift_reg[j][i]);
//							q_pe_in.write(q_shift_reg[j][i]);
//						}
						if (i == 0) {
							i_shift_reg[j][0] = i_shift_reg[j][PE-1];
						} else {
							i_shift_reg[j][i] = i_shift_reg[j][i-1];
						}
					}
				}

			} // end of PE times
//		} // end of Q stream iteration
//	} // end of I stream iteration
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
		i_conv.f = axis_i_in.read()/10.0;
		q_conv.f = axis_q_in.read()/10.0;
		i_in.write(i_conv.f);
		q_in.write(q_conv.f);
	}

	scd_pe(i_in, q_in, i_out, q_out);

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
