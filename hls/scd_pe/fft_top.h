#include "ap_fixed.h"
#include "hls_fft.h"

// configurable params
const char FFT_INPUT_WIDTH                     = 16; // default: 16
const char FFT_OUTPUT_WIDTH                    = FFT_INPUT_WIDTH;
const char FFT_STATUS_WIDTH                    = 8; // Add
const char FFT_CONFIG_WIDTH                    = 16; // default: 16
const char FFT_NFFT_MAX                        = 5; // default: 10
const int  FFT_LENGTH                          = 1 << FFT_NFFT_MAX; // 256
const char PHASE_FACTOR_WIDTH                  = 18; // default: 16

using namespace std;

// FFT Struct Parameters
struct config1 : hls::ip_fft::params_t {
    static const unsigned ordering_opt = hls::ip_fft::natural_order; // bit_reversed_order
    static const unsigned rounding_opt = hls::ip_fft::convergent_rounding; // truncation
//    static const unsigned butterfly_type = hls::ip_fft::use_xtremedsp_slices; // use_luts
    static const unsigned complex_mult_type = hls::ip_fft::use_mults_performance; // use_mults_resources
    static const unsigned arch_opt = hls::ip_fft::radix_2_burst_io; //radix_2_lite_burst_io

    static const unsigned max_nfft = FFT_NFFT_MAX; // 5 (32-point)
    static const unsigned phase_factor_width = PHASE_FACTOR_WIDTH; // 18
    static const unsigned config_width = FFT_CONFIG_WIDTH;
    static const unsigned status_width = FFT_STATUS_WIDTH;
};

typedef hls::ip_fft::config_t<config1> config_t;
typedef hls::ip_fft::status_t<config1> status_t;

// Fixed-point data type
typedef ap_fixed<FFT_INPUT_WIDTH,1> data_t;
typedef ap_fixed<FFT_INPUT_WIDTH,1> data_in_t;
typedef ap_fixed<FFT_OUTPUT_WIDTH,FFT_OUTPUT_WIDTH-FFT_INPUT_WIDTH+1> data_out_t;

#include <complex>
typedef std::complex<data_in_t> cmpxData;
typedef std::complex<data_out_t> cmpxDataOut;

void dummy_proc_fe(
	bool direction,
	config_t* config,
	cmpxData in[FFT_LENGTH],
	cmpxData out[FFT_LENGTH]);

void dummy_proc_be(
    status_t* status_in,
    bool* ovflo,
	cmpxData in[FFT_LENGTH],
	cmpxData out[FFT_LENGTH]);

void fft_top(
	bool direction,
	cmpxData in[FFT_LENGTH],
	cmpxData out[FFT_LENGTH],
	bool* ovflo);
