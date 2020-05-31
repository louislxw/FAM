#include "ap_int.h"
#include "ap_fixed.h"

#define EXP_W 18
#define EXP_I 1
#define IQ_W 16
#define IQ_I 1
#define PI 3.141592654

typedef ap_fixed<18, 1, AP_RND, AP_SAT> exp_fixed_t; // exponential coefficients <18, 1>

/********** Preprocess data type **********/
// i_error_max = 0.000031,  q_error_max = 0.000031, RMSE = 0.000000 (ap_fixed<24,2>, ap_fixed<16,1>)
// i_error_max = 0.000008,  q_error_max = 0.000008, RMSE = 0.000000 (ap_fixed<24,2>, ap_fixed<18,1>)
typedef ap_fixed<16, 1, AP_RND, AP_SAT> single_fixed_t; // Input <16, 1> or <18, 1>

typedef ap_fixed<24, 2, AP_RND, AP_SAT> double_fixed_t; // 256-pt FFT intermediate <32, 2>
typedef ap_fixed<16, 1, AP_RND, AP_SAT> single_fixed_t1; // 256-pt FFT <24, 1>

typedef ap_fixed<24, 2, AP_RND, AP_SAT> double_fixed_t1; // Down Conversion intermediate <32, 2>
typedef ap_fixed<16, 1, AP_RND, AP_SAT> single_fixed_t2; // Down Conversion <24, 1>


/********** SCD Matrix data type **********/
// i_error_max = 0.000031,  q_error_max = 0.000031, RMSE = 0.000000 (ap_fixed<24,2>, ap_fixed<16,1>)
// i_error_max = 0.000008,  q_error_max = 0.000008, RMSE = 0.000000 (ap_fixed<24,2>, ap_fixed<18,1>)
typedef ap_fixed<24, 2, AP_RND, AP_SAT> double_fixed_t2; // Conjugate Mult intermediate <32, 2>
typedef ap_fixed<16, 1, AP_RND, AP_SAT> single_fixed_t3; // Conjugate Mult <24, 1>

typedef ap_fixed<24, 2, AP_RND, AP_SAT> double_fixed_t3; // 32-pt FFT intermediate <32, 2>
typedef ap_fixed<16, 1, AP_RND, AP_SAT> single_fixed_t4; // 32-pt FFT <24, 1>
// verified!
typedef ap_fixed<24, 2, AP_RND, AP_SAT> double_fixed_t4; // alpha_sqr <32, 2>
typedef ap_fixed<16, 1, AP_RND, AP_SAT> single_fixed_t5; // alpha_out <24, 1>


/*** sqrt parameters
#include "hls_dsp.h"
const int DataFormat = hls::CORDIC_FORMAT_USIG_INT;
const int InputWidth = 30;
const int OutputWidth = 16; // Output width for integer square root must be (InputWidth/2)+1
const int RoundMode = hls::CORDIC_ROUND_TRUNCATE;

void sqrt_top(const hls::sqrt_input<InputWidth, DataFormat>::in &x,
              hls::sqrt_output<OutputWidth, DataFormat>::out &sqrtX);
***/
