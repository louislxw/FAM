//*************************************************************************
// Copyright (C) 2018 Kaan Kara - Systems Group, ETH Zurich

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//*************************************************************************

#pragma once
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;
#include <iostream>
using namespace std;
#include <assert.h>

#include <misc.h>

// #define MVU_DEBUG
// #define MVU_DEBUG2

// e.g. usage in MVAU_rowfirst:
//      ACTIVATE<Wbit, Mbit, Mbit, Abit, ScaleBits, FactorScaleBits>(resultVec[p], factorA[p][wMat], factorB[p][wMat]);
template <	unsigned Wbit,
			unsigned Mbit,
			unsigned M2bit,
			unsigned Abit,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
ap_uint<Abit> ACTIVATE(
	ap_int<M2bit> value, 
	ap_int<Mbit> factorA,
	ap_int<Mbit> factorB)
{
#pragma HLS PIPELINE II=1
	const ap_uint<Abit> limit = (1 << Abit)-1;

	ap_uint<Abit> result = 0;

	ap_int<Mbit+M2bit> temp_result = factorA*value;

	if (Wbit > 1)
		temp_result = temp_result >> ScaleBits;

	temp_result = temp_result + factorB;

    // ap_uint<1> remains = temp_result(FactorScaleBits-1, FactorScaleBits-1);
	ap_uint<Abit> remains = temp_result(FactorScaleBits-1, FactorScaleBits-Abit);
	ap_uint<1> roundbit = temp_result(FactorScaleBits-Abit-1, FactorScaleBits-Abit-1);
	temp_result = temp_result >> FactorScaleBits;

    // clip_by_value(0,0.75) needs to be added here
	if (temp_result < 0)
		result = 0;
	// else if (temp_result >= limit)
	else if (temp_result >= 1)
		result = limit;
	else if (remains == limit)
		result = limit;
    else
		result = remains + roundbit;
		// result = temp_result(Abit-1, 0) + remains;

	return result;
}

// ap_int<Mbit> acc = TW_DOT<TWScalebit, Ibit, Mbit, InP>( tempMat, Wp, Wn, tempVec );
// Wbit is always set to 2b
template <	unsigned TWScalebit,
			unsigned Ibit,
			unsigned Mbit,
			unsigned P>
ap_int<Mbit> TW_DOT(
	ap_uint<P*2> weights, 
    ap_uint<TWScalebit> Wp,
    ap_uint<TWScalebit> Wn,
	ap_uint<P*Ibit> in) 
{
	ap_int<Mbit> accumulation = 0;

	for (unsigned p = 0; p < P; p++) {
#pragma HLS UNROLL
		ap_int<Mbit> result;

        ap_uint<Ibit> temp = in( (p+1)*Ibit-1, p*Ibit );
        ap_int<2> weight = weights( (p+1)*2-1, p*2 );
        if (weight == 0) {
            result = 0;
        } else if (weight == -1) {
            result = -((ap_int<Mbit>)temp) * (ap_int<Mbit>)Wn;
        } else {
            result = (ap_int<Mbit>)temp * (ap_int<Mbit>)Wp;
        }

		accumulation += result;
	}

	return accumulation;
}

// specifically for ternary-weights layers, Wbit == 2
template <	unsigned NumVecs,
			unsigned Ibit,
			unsigned TWScalebit,
			unsigned Mbit,
			unsigned Abit,
			unsigned MatrixH,
			unsigned MatrixW,
			unsigned InP,
			unsigned OutP,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
void TW_MVAU_rowfirst(
	stream<ap_uint<InP*Ibit> >& vec, 
	const ap_uint<InP*2> weights[OutP][(MatrixH/InP)*(MatrixW/OutP)],
    const ap_uint<TWScalebit> Wp,
    const ap_uint<TWScalebit> Wn,
	const ap_int<Mbit> factorA[OutP][MatrixW/OutP],
	const ap_int<Mbit> factorB[OutP][MatrixW/OutP],
	stream<ap_uint<OutP*Abit> >& out, 
	const unsigned reps = 1) 
{
	static_assert( MatrixH%InP == 0, "MatrixH mod InP is not 0" );
	static_assert( MatrixW%OutP == 0, "MatrixW mod OutP is not 0");

	const unsigned InputFold = MatrixH/InP;
	const unsigned OutputFold = MatrixW/OutP;

#ifdef MVU_DEBUG
	cout << "InputFold: " << InputFold << endl;
	cout << "OutputFold: " << OutputFold << endl;
#endif

	const unsigned totalReps = reps*NumVecs*InputFold*OutputFold;

	ap_uint<InP*Ibit> rowstore[InputFold];
#pragma HLS RESOURCE variable=rowstore core=RAM_2P_BRAM

	ap_int<Mbit> resultVec[OutP];
#pragma HLS ARRAY_PARTITION variable=resultVec complete dim=0
	unsigned wVec = 0;
	unsigned wMat = 0;
	ap_uint<InP*Ibit> tempVec;
	ap_uint<OutP*Abit> outBuf;

	unsigned index = 0;
	for (unsigned rep = 0; rep < totalReps; rep++) {
#pragma HLS PIPELINE II=1
		
		if (wMat == 0) {
			tempVec = vec.read();
			rowstore[wVec] = tempVec;
		}
		else {
			tempVec = rowstore[wVec];
		}

		index = wVec*OutputFold+wMat;
		for (unsigned p = 0; p < OutP; p++) {
#pragma HLS UNROLL
			ap_uint<InP*2> tempMat = weights[p][index];

			ap_int<Mbit> acc = TW_DOT<TWScalebit, Ibit, Mbit, InP>( tempMat, Wp, Wn, tempVec );

			if (wVec == 0)
				resultVec[p] = acc;
			else
				resultVec[p] += acc;
            
			outBuf( (p+1)*Abit-1, p*Abit ) = ACTIVATE<2, Mbit, Mbit, Abit, ScaleBits, FactorScaleBits>(resultVec[p], factorA[p][wMat], factorB[p][wMat]);
		}

		if (wVec == InputFold-1){
			out.write(outBuf);
		}

		if (wVec == InputFold-1) {
			wVec = 0;
			if (wMat == OutputFold-1)
				wMat = 0;
			else
				wMat++;
		}
		else
			wVec++;
	}
}

template <	unsigned Wbit,
			unsigned Ibit,
			unsigned Mbit,
			unsigned P>
ap_int<Mbit> DOT(
	ap_uint<P*Wbit> weights, 
	ap_uint<P*Ibit> in) 
{	
	ap_int<Mbit> accumulation = 0;

	for (unsigned p = 0; p < P; p++) {
#pragma HLS UNROLL
		ap_int<Mbit> result;

		if (Wbit == 1) {
			ap_uint<Ibit> temp = in( (p+1)*Ibit-1, p*Ibit );
			if (weights(p,p) == 0)
				result = temp;
			else
				result = -temp;
		}
		else {
			ap_int<Wbit> temp_w = weights( (p+1)*Wbit-1, p*Wbit );
			ap_int<Ibit+1> temp_in = in( (p+1)*Ibit-1, p*Ibit );
			result = temp_w*temp_in;
		}

		accumulation += result;
	}

	return accumulation;
}

template <	unsigned NumVecs,
			unsigned Ibit,
			unsigned TWScalebit,
			unsigned Mbit,
			unsigned Abit,
			unsigned MatrixH,
			unsigned MatrixW,
			unsigned InP,
			unsigned OutP,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
void TW_MVAU(
	stream<ap_uint<InP*Ibit> >& vec, 
	const ap_uint<InP*2> weights[OutP][(MatrixH/InP)*(MatrixW/OutP)],
    const ap_int<TWScalebit> Wp,
    const ap_int<TWScalebit> Wn,
	const ap_int<Mbit> factorA[OutP][MatrixW/OutP],
	const ap_int<Mbit> factorB[OutP][MatrixW/OutP],
	stream<ap_uint<OutP*Abit> >& out, 
	const unsigned reps = 1) 
{
	static_assert( MatrixH%InP == 0, "MatrixH mod InP is not 0" );
	static_assert( MatrixW%OutP == 0, "MatrixW mod OutP is not 0");

	const unsigned InputFold = MatrixH/InP;
	const unsigned OutputFold = MatrixW/OutP;

#ifdef MVU_DEBUG
	cout << "InputFold: " << InputFold << endl;
	cout << "OutputFold: " << OutputFold << endl;
#endif

	const unsigned totalReps = reps*NumVecs*InputFold*OutputFold;

	ap_int<Mbit> resultVec[OutP][OutputFold];
#pragma HLS ARRAY_PARTITION variable=resultVec complete dim=0
	unsigned wVec = 0;
	unsigned wMat = 0;
	ap_uint<InP*Ibit> tempVec;
	ap_uint<OutP*Abit> outBuf;

	unsigned index = 0;
	for (unsigned rep = 0; rep < totalReps; rep++) {
#pragma HLS PIPELINE II=1
		
		if (wMat == 0) {
			tempVec = vec.read();
		}

		index = wVec*OutputFold+wMat;
		for (unsigned p = 0; p < OutP; p++) {
#pragma HLS UNROLL
			ap_uint<InP*2> tempMat = weights[p][index];

			ap_int<Mbit> acc = DOT<TWScalebit, Ibit, Mbit, InP>( tempMat, Wp, Wn, tempVec );

			if (wVec == 0)
				resultVec[p][wMat] = acc;
			else
				resultVec[p][wMat] += acc;

			outBuf( (p+1)*Abit-1, p*Abit ) = ACTIVATE<2, Mbit, Mbit, Abit, ScaleBits, FactorScaleBits>(resultVec[p][wMat], factorA[p][wMat], factorB[p][wMat]);
		}

		if (wVec == InputFold-1){
			out.write(outBuf);
		}

		if (wMat == OutputFold-1) {
			wMat = 0;
			if (wVec == InputFold-1)
				wVec = 0;
			else
				wVec++;
		}
		else
			wMat++;
	}
}

template <	unsigned NumVecs,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned Abit,
			unsigned MatrixH,
			unsigned MatrixW,
			unsigned InP,
			unsigned OutP,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
void MVAU(
	stream<ap_uint<InP*Ibit> >& vec, 
	const ap_uint<InP*Wbit> weights[OutP][(MatrixH/InP)*(MatrixW/OutP)],
	const ap_int<Mbit> factorA[OutP][MatrixW/OutP],
	const ap_int<Mbit> factorB[OutP][MatrixW/OutP],
	stream<ap_uint<OutP*Abit> >& out, 
	const unsigned reps = 1) 
{
	static_assert( MatrixH%InP == 0, "MatrixH mod InP is not 0" );
	static_assert( MatrixW%OutP == 0, "MatrixW mod OutP is not 0");

	const unsigned InputFold = MatrixH/InP;
	const unsigned OutputFold = MatrixW/OutP;

#ifdef MVU_DEBUG
	cout << "InputFold: " << InputFold << endl;
	cout << "OutputFold: " << OutputFold << endl;
#endif

	const unsigned totalReps = reps*NumVecs*InputFold*OutputFold;

	ap_int<Mbit> resultVec[OutP][OutputFold];
#pragma HLS ARRAY_PARTITION variable=resultVec complete dim=0
	unsigned wVec = 0;
	unsigned wMat = 0;
	ap_uint<InP*Ibit> tempVec;
	ap_uint<OutP*Abit> outBuf;

	unsigned index = 0;
	for (unsigned rep = 0; rep < totalReps; rep++) {
#pragma HLS PIPELINE II=1
		
		if (wMat == 0) {
			tempVec = vec.read();
		}

		index = wVec*OutputFold+wMat;
		for (unsigned p = 0; p < OutP; p++) {
#pragma HLS UNROLL
			ap_uint<InP*Wbit> tempMat = weights[p][index];

			ap_int<Mbit> acc = DOT<Wbit, Ibit, Mbit, InP>( tempMat, tempVec );

			if (wVec == 0)
				resultVec[p][wMat] = acc;
			else
				resultVec[p][wMat] += acc;

			outBuf( (p+1)*Abit-1, p*Abit ) = ACTIVATE<Wbit, Mbit, Mbit, Abit, ScaleBits, FactorScaleBits>(resultVec[p][wMat], factorA[p][wMat], factorB[p][wMat]);
		}

		if (wVec == InputFold-1){
			out.write(outBuf);
		}

		if (wMat == OutputFold-1) {
			wMat = 0;
			if (wVec == InputFold-1)
				wVec = 0;
			else
				wVec++;
		}
		else
			wMat++;
	}
}

template <	unsigned NumVecs,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned Abit,
			unsigned MatrixH,
			unsigned MatrixW,
			unsigned InP,
			unsigned OutP,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
void MVAU_rowfirst(
	stream<ap_uint<InP*Ibit> >& vec, 
	const ap_uint<InP*Wbit> weights[OutP][(MatrixH/InP)*(MatrixW/OutP)],
	const ap_int<Mbit> factorA[OutP][MatrixW/OutP],
	const ap_int<Mbit> factorB[OutP][MatrixW/OutP],
	stream<ap_uint<OutP*Abit> >& out, 
	const unsigned reps = 1) 
{
	static_assert( MatrixH%InP == 0, "MatrixH mod InP is not 0" );
	static_assert( MatrixW%OutP == 0, "MatrixW mod OutP is not 0");

	const unsigned InputFold = MatrixH/InP;
	const unsigned OutputFold = MatrixW/OutP;

#ifdef MVU_DEBUG
	cout << "InputFold: " << InputFold << endl;
	cout << "OutputFold: " << OutputFold << endl;
#endif

	const unsigned totalReps = reps*NumVecs*InputFold*OutputFold;

	ap_uint<InP*Ibit> rowstore[InputFold];
#pragma HLS RESOURCE variable=rowstore core=RAM_2P_BRAM

	ap_int<Mbit> resultVec[OutP];
#pragma HLS ARRAY_PARTITION variable=resultVec complete dim=0
	unsigned wVec = 0;
	unsigned wMat = 0;
	ap_uint<InP*Ibit> tempVec;
	ap_uint<OutP*Abit> outBuf;

	unsigned index = 0;
	for (unsigned rep = 0; rep < totalReps; rep++) {
#pragma HLS PIPELINE II=1
		
		if (wMat == 0) {
			tempVec = vec.read();
			rowstore[wVec] = tempVec;
		}
		else {
			tempVec = rowstore[wVec];
		}

		index = wVec*OutputFold+wMat;
		for (unsigned p = 0; p < OutP; p++) {
#pragma HLS UNROLL
			ap_uint<InP*Wbit> tempMat = weights[p][index];

			ap_int<Mbit> acc = DOT<Wbit, Ibit, Mbit, InP>( tempMat, tempVec );

			if (wVec == 0)
				resultVec[p] = acc;
			else
				resultVec[p] += acc;
            
			outBuf( (p+1)*Abit-1, p*Abit ) = ACTIVATE<Wbit, Mbit, Mbit, Abit, ScaleBits, FactorScaleBits>(resultVec[p], factorA[p][wMat], factorB[p][wMat]);
		}

		if (wVec == InputFold-1){
			out.write(outBuf);
		}

		if (wVec == InputFold-1) {
			wVec = 0;
			if (wMat == OutputFold-1)
				wMat = 0;
			else
				wMat++;
		}
		else
			wVec++;
	}
}

template <	unsigned NumVecs,
			unsigned Ibit,
			unsigned TWScalebit,
			unsigned TWFracbit,
			unsigned Mbit,
			unsigned MatrixH,
			unsigned MatrixW,
			unsigned InP,
			unsigned OutP>
void TW_MVU_rowfirst(
	stream<ap_uint<InP*Ibit> >& vec, 
	const ap_uint<InP*2> weights[OutP][(MatrixH/InP)*(MatrixW/OutP)], 
    const ap_uint<TWScalebit> Wp,
    const ap_uint<TWScalebit> Wn,
	stream<ap_uint<OutP*Mbit> >& out, 
	const unsigned reps = 1) 
{
	static_assert( MatrixH%InP == 0, "MatrixH mod InP is not 0" );
	static_assert( MatrixW%OutP == 0, "MatrixW mod OutP is not 0");

	const unsigned InputFold = MatrixH/InP;
	const unsigned OutputFold = MatrixW/OutP;

#ifdef MVU_DEBUG
	cout << "InputFold: " << InputFold << endl;
	cout << "OutputFold: " << OutputFold << endl;
#endif

	const unsigned totalReps = reps*NumVecs*InputFold*OutputFold;

	ap_uint<InP*Ibit> rowstore[InputFold];
#pragma HLS RESOURCE variable=rowstore core=RAM_2P_BRAM

	ap_uint<Mbit> resultVec[OutP];
#pragma HLS ARRAY_PARTITION variable=resultVec complete dim=0
	unsigned wVec = 0;
	unsigned wMat = 0;
	ap_uint<InP*Ibit> tempVec;
	ap_uint<OutP*Mbit> outBuf;
    int pixnum = -1;

	unsigned index = 0;
	for (unsigned rep = 0; rep < totalReps; rep++) {
#pragma HLS PIPELINE II=1
		
		if (wMat == 0) {
			tempVec = vec.read();
			rowstore[wVec] = tempVec;
            pixnum++;
		}
		else {
			tempVec = rowstore[wVec];
		}

		index = wVec*OutputFold+wMat;
		for (unsigned p = 0; p < OutP; p++) {
#pragma HLS UNROLL
			ap_uint<InP*2> tempMat = weights[p][index];
			
			ap_int<Mbit> acc = TW_DOT<TWScalebit, Ibit, Mbit, InP>( tempMat, Wp, Wn, tempVec );

            ap_int<Mbit> temp_result = acc;
            ap_uint<1> roundbit = temp_result(TWFracbit-1, TWFracbit-1);
            temp_result = temp_result >> TWFracbit;
            temp_result += roundbit;

			if (wVec == 0)
				resultVec[p] = temp_result;
			else
				resultVec[p] += temp_result;

			outBuf((p+1)*Mbit-1, p*Mbit) = resultVec[p];
		}

		if (wVec == InputFold-1)
			out.write(outBuf);

		if (wVec == InputFold-1) {
			wVec = 0;
			if (wMat == OutputFold-1)
				wMat = 0;
			else
				wMat++;
		}
		else
			wVec++;
	}
}

template <	unsigned NumVecs,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned MatrixH,
			unsigned MatrixW,
			unsigned InP,
			unsigned OutP>
void MVU_rowfirst(
	stream<ap_uint<InP*Ibit> >& vec, 
	const ap_uint<InP*Wbit> weights[OutP][(MatrixH/InP)*(MatrixW/OutP)], 
	stream<ap_uint<OutP*Mbit> >& out, 
	const unsigned reps = 1) 
{
	static_assert( MatrixH%InP == 0, "MatrixH mod InP is not 0" );
	static_assert( MatrixW%OutP == 0, "MatrixW mod OutP is not 0");

	const unsigned InputFold = MatrixH/InP;
	const unsigned OutputFold = MatrixW/OutP;

#ifdef MVU_DEBUG
	cout << "InputFold: " << InputFold << endl;
	cout << "OutputFold: " << OutputFold << endl;
#endif

	const unsigned totalReps = reps*NumVecs*InputFold*OutputFold;

	ap_uint<InP*Ibit> rowstore[InputFold];
#pragma HLS RESOURCE variable=rowstore core=RAM_2P_BRAM

	ap_uint<Mbit> resultVec[OutP];
#pragma HLS ARRAY_PARTITION variable=resultVec complete dim=0
	unsigned wVec = 0;
	unsigned wMat = 0;
	ap_uint<InP*Ibit> tempVec;
	ap_uint<OutP*Mbit> outBuf;

	unsigned index = 0;
	for (unsigned rep = 0; rep < totalReps; rep++) {
#pragma HLS PIPELINE II=1
		
		if (wMat == 0) {
			tempVec = vec.read();
			rowstore[wVec] = tempVec;
		}
		else {
			tempVec = rowstore[wVec];
		}

		index = wVec*OutputFold+wMat;
		for (unsigned p = 0; p < OutP; p++) {
#pragma HLS UNROLL
			ap_uint<InP*Wbit> tempMat = weights[p][index];
			
			ap_int<Mbit> acc = DOT<Wbit, Ibit, Mbit, InP>( tempMat, tempVec );

			if (wVec == 0)
				resultVec[p] = acc;
			else
				resultVec[p] += acc;

			outBuf((p+1)*Mbit-1, p*Mbit) = resultVec[p];
		}

		if (wVec == InputFold-1)
			out.write(outBuf);

		if (wVec == InputFold-1) {
			wVec = 0;
			if (wMat == OutputFold-1)
				wMat = 0;
			else
				wMat++;
		}
		else
			wVec++;
	}
}

template <	unsigned Ibit,
			unsigned TWScalebit,
			unsigned Mbit,
			unsigned Abit,
			unsigned MAX_MatrixH,
			unsigned MAX_MatrixW,
			unsigned InP,
			unsigned OutP,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
void TW_MVAU_variable(
	stream<ap_uint<InP*Ibit> >& vec, 
	const ap_uint<InP*2> weights[OutP][(MAX_MatrixH/InP)*(MAX_MatrixW/OutP)],
    const ap_uint<TWScalebit> Wp,
    const ap_uint<TWScalebit> Wn,
	const ap_int<Mbit> factorA[OutP][MAX_MatrixW/OutP],
	const ap_int<Mbit> factorB[OutP][MAX_MatrixW/OutP],
	stream<ap_uint<OutP*Abit> >& out,
	const unsigned NumVecs,
	// const unsigned MatrixH,
	// const unsigned MatrixW,
	const unsigned reps = 1) 
{
	const unsigned MAX_OutputFold = MAX_MatrixW/OutP;
	// const unsigned InputFold = MatrixH/InP;
	// const unsigned OutputFold = MatrixW/OutP;
	const unsigned InputFold = MAX_MatrixH/InP;
	const unsigned OutputFold = MAX_MatrixW/OutP;

	const unsigned M2bit = 2+12; // originally was (Wbit + 12), this hardcode might have introduced buggy behaviour

#ifdef MVU_DEBUG
	cout << "InputFold: " << InputFold << endl;
	cout << "OutputFold: " << OutputFold << endl;
#endif

	const unsigned totalReps = reps*NumVecs*InputFold*OutputFold;

	ap_uint<InP*Ibit> rowstore[InputFold];
#pragma HLS RESOURCE variable=rowstore core=RAM_2P_BRAM

	ap_uint<M2bit> resultVec[OutP];
#pragma HLS ARRAY_PARTITION variable=resultVec complete dim=0
	unsigned wVec = 0;
	unsigned wMat = 0;
	ap_uint<InP*Ibit> tempVec;
	ap_uint<OutP*Abit> outBuf;

	for (unsigned rep = 0; rep < totalReps; rep++) {
#pragma HLS PIPELINE II=1
		
		if (wMat == 0) {
			tempVec = vec.read();
			rowstore[wVec] = tempVec;
		}
		else {
			tempVec = rowstore[wVec];
		}
		
		for (unsigned p = 0; p < OutP; p++) {
#pragma HLS UNROLL
			ap_uint<InP*2> tempMat = weights[p][wVec*OutputFold+wMat];

			ap_int<M2bit> acc = TW_DOT<TWScalebit, Ibit, M2bit, InP>( tempMat, Wp, Wn, tempVec );

			if (wVec == 0)
				resultVec[p] = acc;
			else
				resultVec[p] += acc;

			outBuf( (p+1)*Abit-1, p*Abit ) = ACTIVATE<2, Mbit, M2bit, Abit, ScaleBits, FactorScaleBits>(resultVec[p], factorA[p][wMat], factorB[p][wMat]);
		}

		if (wVec == InputFold-1){
			out.write(outBuf);
		}

		if (wVec == InputFold-1) {
			wVec = 0;
			if (wMat == OutputFold-1)
				wMat = 0;
			else
				wMat++;
		}
		else
			wVec++;
	}
}

template <	unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned Abit,
			unsigned MAX_MatrixH,
			unsigned MAX_MatrixW,
			unsigned InP,
			unsigned OutP,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
void MVAU_variable(
	stream<ap_uint<InP*Ibit> >& vec, 
	const ap_uint<InP*Wbit> weights[OutP][(MAX_MatrixH/InP)*(MAX_MatrixW/OutP)],
	const ap_int<Mbit> factorA[OutP][MAX_MatrixW/OutP],
	const ap_int<Mbit> factorB[OutP][MAX_MatrixW/OutP],
	stream<ap_uint<OutP*Abit> >& out,
	const unsigned NumVecs,
	// const unsigned MatrixH,
	// const unsigned MatrixW,
	const unsigned reps = 1) 
{
	const unsigned MAX_OutputFold = MAX_MatrixW/OutP;
	// const unsigned InputFold = MatrixH/InP;
	// const unsigned OutputFold = MatrixW/OutP;
	const unsigned InputFold = MAX_MatrixH/InP;
	const unsigned OutputFold = MAX_MatrixW/OutP;

	const unsigned M2bit = Wbit+12;

#ifdef MVU_DEBUG
	cout << "InputFold: " << InputFold << endl;
	cout << "OutputFold: " << OutputFold << endl;
#endif

	const unsigned totalReps = reps*NumVecs*InputFold*OutputFold;

	ap_uint<InP*Ibit> rowstore[InputFold];
#pragma HLS RESOURCE variable=rowstore core=RAM_2P_BRAM

	ap_uint<M2bit> resultVec[OutP];
#pragma HLS ARRAY_PARTITION variable=resultVec complete dim=0
	unsigned wVec = 0;
	unsigned wMat = 0;
	ap_uint<InP*Ibit> tempVec;
	ap_uint<OutP*Abit> outBuf;

	for (unsigned rep = 0; rep < totalReps; rep++) {
#pragma HLS PIPELINE II=1
		
		if (wMat == 0) {
			tempVec = vec.read();
			rowstore[wVec] = tempVec;
		}
		else {
			tempVec = rowstore[wVec];
		}
		
		for (unsigned p = 0; p < OutP; p++) {
#pragma HLS UNROLL
			ap_uint<InP*Wbit> tempMat = weights[p][wVec*OutputFold+wMat];

			ap_int<M2bit> acc = DOT<Wbit, Ibit, M2bit, InP>( tempMat, tempVec );

			if (wVec == 0)
				resultVec[p] = acc;
			else
				resultVec[p] += acc;

			outBuf( (p+1)*Abit-1, p*Abit ) = ACTIVATE<Wbit, Mbit, M2bit, Abit, ScaleBits, FactorScaleBits>(resultVec[p], factorA[p][wMat], factorB[p][wMat]);
		}

		if (wVec == InputFold-1){
			out.write(outBuf);
		}

		if (wVec == InputFold-1) {
			wVec = 0;
			if (wMat == OutputFold-1)
				wMat = 0;
			else
				wMat++;
		}
		else
			wVec++;
	}
}

template <	unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned MAX_MatrixW,
			unsigned InP,
			unsigned OutP>
void MVU_variable(
	stream<ap_uint<InP*Ibit> >& vec, 
	const ap_uint<InP*Wbit>* weights[OutP],
	stream<ap_uint<OutP*Mbit> >& out,
	unsigned NumVecs,
	unsigned MatrixH,
	unsigned MatrixW,
	const unsigned reps = 1) 
{
	const unsigned MAX_OutputFold = MAX_MatrixW/OutP;
	unsigned InputFold = MatrixH/InP;
	unsigned OutputFold = MatrixW/OutP;

#ifdef MVU_DEBUG
	cout << "InputFold: " << InputFold << endl;
	cout << "OutputFold: " << OutputFold << endl;
#endif

	const unsigned totalReps = reps*NumVecs;

	ap_uint<Mbit> resultVec[OutP][MAX_OutputFold];
#pragma HLS ARRAY_PARTITION variable=resultVec complete dim=0
	unsigned wVec = 0;
	unsigned wMat = 0;
	ap_uint<InP*Ibit> tempVec;
	ap_uint<OutP*Mbit> outBuf;

	for (unsigned rep = 0; rep < totalReps*InputFold*OutputFold; rep++) {
#pragma HLS PIPELINE II=1
		
		if (wMat == 0)
			tempVec = vec.read();

		for (unsigned p = 0; p < OutP; p++) {
#pragma HLS UNROLL
			ap_uint<InP*Wbit> tempMat = weights[p][wVec*OutputFold+wMat];

			ap_int<Mbit> acc = DOT<Wbit, Ibit, Mbit, InP>( tempMat, tempVec );

			if (wVec == 0)
				resultVec[p][wMat] = acc;
			else
				resultVec[p][wMat] += acc;

			outBuf( (p+1)*Mbit-1, p*Mbit ) = resultVec[p][wMat];
		}

		if (wVec == InputFold-1){
			out.write(outBuf);
		}

		if (wMat == OutputFold-1) {
			wMat = 0;
			if (wVec == InputFold-1)
				wVec = 0;
			else
				wVec++;
		}
		else
			wMat++;
	}
}
