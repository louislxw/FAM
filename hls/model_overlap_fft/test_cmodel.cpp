#include <stdio.h>
#include <stdlib.h>
#include "hls_stream.h"
#include "ap_int.h"
#include "hls-nn-lib.h"
#include "cmodel_weights.h"

using namespace hls;
using namespace std;

typedef ap_uint<64> CONV1_OUT_T;
typedef ap_uint<64> POOL1_OUT_T;
typedef ap_uint<40> FIRE1_OUT_T;
typedef ap_uint<40> POOL2_OUT_T;
typedef ap_uint<40> FIRE2_OUT_T;
typedef ap_uint<40> POOL3_OUT_T;
typedef ap_uint<40> FIRE3_OUT_T;
typedef ap_int<16> CONV_CLASS_OUT_T;
typedef ap_uint<32> PREDS_OUT_T;
typedef ap_uint<8> PIXEL_T;
typedef ap_uint<1> PRED_T;

#define IorQ_W 16
#define FFT_W 32
#define PIXEL_W 8

struct axis_in{
	ap_uint<8> data;
	bool last;
};
struct axis_out{
	ap_uint<8> data;
	bool last;
};
typedef hls::stream<axis_in> AXIS_IN;
typedef hls::stream<axis_out> AXIS_OUT;

void model_wrapper(
		stream<ap_int<IorQ_W> >& i_in,
		stream<ap_int<IorQ_W> >& q_in,
		AXIS_OUT &axis_pred_out
);

int main(int argc, char **argv) {
	printf("Hello, World!\n");
	stream<ap_int<IorQ_W> > i_in("i_in_stream");
	stream<ap_int<IorQ_W> > q_in("q_in_stream");
	AXIS_OUT axis_image_out("image_out_stream");
	AXIS_OUT axis_pred_out("pred_out_stream");
	axis_in stream_i_in, stream_q_in;
	axis_out pred_out;

	ap_int<32> in_size = 64*319; // 1024*1024
	// ap_int<32> image_size = 64*64;
	ap_int<32> out_size = 256; // 256

	for (int i = 0; i < in_size; i++) {
		stream_i_in.data = (ap_int<16>(i%256));
		stream_q_in.data = (ap_int<16>(i%256));
		// i_in << stream_i_in.data;
		// q_in << stream_q_in.data;
		i_in.write(stream_i_in.data);
		q_in.write(stream_q_in.data);
	}

	model_wrapper(i_in, q_in, axis_pred_out);

    for (int i = 0; i < out_size; i++) {
    	pred_out = axis_pred_out.read();

        printf("pred_out[%d] = %d, TLAST = %d\n", i, (uint32_t)pred_out.data, (unsigned)pred_out.last);
    }

	return 0;
}
