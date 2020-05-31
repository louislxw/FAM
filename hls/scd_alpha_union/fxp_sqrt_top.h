#ifndef FXP_SQRT_TOP_H_
#define FXP_SQRT_TOP_H_

#include "fxp_sqrt.h"
#include <ap_int.h>

#define IN_BW   24 // 24
#define IN_IW    2 // 8
#define OUT_BW  16 // 28
#define OUT_IW   1 // ((IN_IW + 1) / 2) // 4

// typedefs for top-level input and output fixed-point formats
typedef ap_ufixed<IN_BW,IN_IW>   in_data_t;
typedef ap_ufixed<OUT_BW,OUT_IW> out_data_t;
//typedef ap_ufixed<IN_BW, IN_IW, AP_RND, AP_SAT>   in_data_t;
//typedef ap_ufixed<OUT_BW, OUT_IW, AP_RND, AP_SAT> out_data_t;

// Top level wrapper function - calls the core template function w/ above types
out_data_t fxp_sqrt_top(in_data_t& in_val);

#endif // FXP_SQRT_TOP_H_ not defined

