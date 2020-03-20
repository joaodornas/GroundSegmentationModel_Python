
#define MAT_ACCESS_col_major(A, row, col, N) A[col*N + row]
#define MAT_ACCESS_row_major(A, row, col, N) A[row*N + col]

/////////////////////////////////////////
// POLAR GRID - GET OP AND SECTORS
/////////////////////////////////////////

//STEP: PolarGrid_getGetOPSectorsAndBins
//OP[i] = sqrt(X[i]^2 + Y[i]^2);
//int sign; int H;
//if (Y[i] > 0) { sign = 1; };
//if (Y[i] < 0) { sign = -1; };
//if (Y[i] == 0) { sign = 0; };
//if (-Y[i] > 0) { H = 0; };
//if (-Y[i] < 0) { H = 1; };
//if (-Y[i] == 0) { H = 1 / 2; };
//SECTOR = round(((sign * acos(X[i]) / OP[i]) + (2 * _VELODYNEOPTIONS.PI * H)) / _VELODYNEOPTIONS.alpha);
kernel void PolarGrid_getOPSectorsAndBins (
    global const float * restrict X,
	global const float * restrict Y,
	global float * restrict OP,
	global int * restrict Sectors,
	global int * restrict Bins,
	float PI,
	float M,
	float R,
	float Rstep,
	global float * restrict _cos,
	global float * restrict _acos,
	global float * restrict ret,
	global float * restrict x
	
)

{ 
	int Aind = get_global_id(0);
	//int Bind = get_group_id(1) + get_local_id(1);

	//CALCULATES OP
	OP[Aind] = sqrt(X[Aind] * X[Aind] + Y[Aind] * Y[Aind]);

	//CALCULATES SECTOR FOR EACH 3D Data POINT
	float _zero = 0.f;
	float _minus_one = -1.f;
	float _sign = 0.0f;
	float negate = 0.f;
	int _exclude = 0;
	_cos[Aind] = X[Aind] / OP[Aind];
	float _alpha = (2 * PI)/M;
	float _H = 0.0f;

	//CALCULATES ANGLE
	x[Aind] = _cos[Aind];
	if (x[Aind] < _zero) { negate = 1.f; }; 
	if (x[Aind] > _zero) { negate = 0.f; };
	if (x[Aind] < _zero) { x[Aind] = _minus_one * x[Aind]; }; 
	ret[Aind] = (float)-0.0187293f;
	ret[Aind] = ret[Aind] * x[Aind];
	ret[Aind] = ret[Aind] + (float)0.0742610f;
	ret[Aind] = ret[Aind] * x[Aind];
	ret[Aind] = ret[Aind] - (float)0.2121144f;
	ret[Aind] = ret[Aind] * x[Aind];
	ret[Aind] = ret[Aind] + (float)1.5707288f;
	ret[Aind] = ret[Aind] * sqrt(1.0f-x[Aind]);
	ret[Aind] = ret[Aind] - 2 * negate * ret[Aind];
	_acos[Aind] = negate * PI + ret[Aind];

	if (Y[Aind] > _zero) { _sign = 1.0f; };
	if (Y[Aind] < _zero) { _sign = -1.0f; };
	if (Y[Aind] == _zero) { _sign = 0.0f; };

	if ((_minus_one * Y[Aind]) > _zero) { _H = 1.0f; };
	if ((_minus_one * Y[Aind]) < _zero) { _H = 0.0f; };
	if ((_minus_one * Y[Aind]) == _zero) { _H = 0.5f; };

	if (OP[Aind] > R) { _exclude = 0; };
	if (OP[Aind] <= R) { _exclude = 1; };

	//SECTOR
	Sectors[Aind] = (int)ceil( ( ( ( _sign * _acos[Aind] ) + (2 * PI * _H) ) / _alpha ) )*_exclude;

	//BINS
	Bins[Aind] = (int)ceil(OP[Aind] / Rstep)*_exclude;

}





/////////////////////////////////////////
// SEED - COPY SP TO SP_TMP
/////////////////////////////////////////

//STEP: Copy Sp to temporary Sp_tmp
//Sp_tmp = Sp; //Sp = Sp U Snew;
kernel void Seed_copySp (
    global float * restrict Sp_Z,
	global float * restrict Sp_OP,
	global int * restrict Sp_Bins,
	global int * restrict Sp_original_idx,
	global float * restrict Sp_Z_tmp,
	global float * restrict Sp_OP_tmp,
	global int * restrict Sp_Bins_tmp,
	global int * restrict Sp_original_idx_tmp
)

{ 
	int Aind = get_global_id(0) ;

	Sp_Z_tmp[Aind] = Sp_Z[Aind];
	Sp_OP_tmp[Aind] = Sp_OP[Aind];
	Sp_Bins_tmp[Aind] = Sp_Bins[Aind];
	Sp_original_idx_tmp[Aind] = Sp_original_idx[Aind];

}

/////////////////////////////////////////
// SEED - GROUP SP U SNEW - STEP 1
/////////////////////////////////////////

//STEP: Sp <- Sp_tmp (Sp = Sp U Snew)
kernel void Seed_SpUSnew_Step1 (
   global float * restrict Sp_Z_tmp,
	global float * restrict Sp_OP_tmp,
	global int * restrict Sp_Bins_tmp,
	global int * restrict Sp_original_idx_tmp,
	global float * restrict Sp_Z,
	global float * restrict Sp_OP,
	global int * restrict Sp_Bins,
	global int * restrict Sp_original_idx
)

{ 
	int Aind = get_global_id(0);

	Sp_Z[Aind] = Sp_Z_tmp[Aind];
	Sp_OP[Aind] = Sp_OP_tmp[Aind];
	Sp_Bins[Aind] = Sp_Bins_tmp[Aind];
	Sp_original_idx[Aind] = Sp_original_idx_tmp[Aind];

}

/////////////////////////////////////////
// SEED - GROUP SP U SNEW - STEP 2
/////////////////////////////////////////

//STEP: Sp <- Snew (Sp = Sp U Snew)
kernel void Seed_SpUSnew_Step2 (
    global float * restrict Snew_Z,
	global float * restrict Snew_OP,
	global int * restrict Snew_Bins,
	global int * restrict Snew_original_idx,
	global float * restrict Sp_Z,
	global float * restrict Sp_OP,
	global int * restrict Sp_Bins,
	global int * restrict Sp_original_idx,
	global const int * restrict start_idx
)

{ 
	int Aind = get_global_id(0);
	int Bind = Aind + start_idx[0];

	Sp_Z[Bind] = Snew_Z[Aind];
	Sp_OP[Bind] = Snew_OP[Aind];
	Sp_Bins[Bind] = Snew_Bins[Aind];
	Sp_original_idx[Bind] = Snew_original_idx[Aind];

}



//////////////////////////////////////
// TEST - EXTRACT TEST DATA POINTS
//////////////////////////////////////
kernel void Test_getTest_Step1(
	global float * restrict Test_Z,
	global float * restrict Test_OP,
	global int * restrict Test_Bins,
	global int * restrict Test_original_idx,
	global float * restrict PGi_Z,
	global float * restrict PGi_OP,
	global int * restrict PGi_Bins,
	global int * restrict PGi_original_idx,
	global int * restrict Sp_original_idx,
	int Sp_dim,
	global int * restrict RemoveCode
        
)
{
    
    int Aind = get_global_id(0);
    //int Bind = get_global_id(1);
    
	int Sp_idx = 0;
	int PGi_idx = 0;
	int shouldTest = true;

	PGi_idx = PGi_original_idx[Aind];

	for ( int i = 0; i < Sp_dim; i++ )
    { 
		Sp_idx = Sp_original_idx[i];

		if (PGi_idx == Sp_idx)
		{
			shouldTest = false;
		}

	}

	float Z = 0;
	float OP = 0;
	int Bins = 0;
	int idx = 0;

	float _RemoveCode_float = (float)RemoveCode[0];
	int _RemoveCode_int = RemoveCode[0];

	if (shouldTest)
	{
		Z = PGi_Z[Aind];
		OP = PGi_OP[Aind];
		Bins = PGi_Bins[Aind];
		idx = PGi_original_idx[Aind];
	}
	else
	{
		Z = _RemoveCode_float;
		OP = _RemoveCode_float;
		Bins = _RemoveCode_int;
		idx = _RemoveCode_int;
	}

	Test_Z[Aind] = Z;
	Test_OP[Aind] = OP;
	Test_Bins[Aind] = Bins;
	Test_original_idx[Aind] = idx;

}

//////////////////////////////////////
// TEST - EXTRACT TEST DATA POINTS
//////////////////////////////////////
kernel void Test_getTest_Step2(
	global float * restrict Test_Z,
	global float * restrict Test_OP,
	global int * restrict Test_Bins,
	global int * restrict Test_original_idx,
	global float * restrict Test_tmp_Z,
	global float * restrict Test_tmp_OP,
	global int * restrict Test_tmp_Bins,
	global int * restrict Test_tmp_original_idx,
	int Test_tmp_dim,
	global int * restrict idxs
        
)
{
    
    int Aind = get_global_id(0);
    //int Bind = get_global_id(1);
    
	int Test_tmp_idx = idxs[Aind];

	Test_Z[Aind] = Test_tmp_Z[Test_tmp_idx];
	Test_OP[Aind] = Test_tmp_OP[Test_tmp_idx];
	Test_Bins[Aind] = Test_tmp_Bins[Test_tmp_idx];
	Test_original_idx[Aind] = Test_tmp_original_idx[Test_tmp_idx];
	

}


//////////////////////////////////////
// EVAL - MATRIX MULTIPLY 
//////////////////////////////////////
kernel void GEMM_matrixMultiply (
    global const float * restrict A,
    int _rows_A,   
    global const float * restrict B,
    int _cols_B,  
    global float * restrict C,
    int _dim_K     
)
{
    
    int Aind = get_global_id(0);
    int Bind = get_global_id(1);
    int Cind = Aind + Bind*_rows_A;

	if ( (Aind < _rows_A) && (Bind < _cols_B) )
    { 
		float sum = 0;

		for (int i = 0; i < _dim_K; i++)
		{
			sum += A[i*_rows_A + Aind] * B[i + Bind*_dim_K];
		}

		C[Cind] = sum;

	}

}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// GAUSS PROCESS - Square Exponential Kernel 
//
// K_X_star = (sigma_f_) * exp( ( -1 / (2 * l_scale_ * l_scale_) ) * ( (X - star)*(X - star) ) );
//
///////////////////////////////////////////////////////////////////////////////////////////////////

//K_X_star = (sigma_f_) * exp( ( -1 / (2 * l_scale_ * l_scale_) ) * ( (X - star)*(X - star) ) );
kernel void Gauss_SquareExponentialKernel (
    global const float * restrict X,
	int _dim_size_X,
	global const float * restrict star,
	int _dim_size_star,
	global float * restrict K_X_star,
	float sigma_f,
	float length_scale
)

{ 
	int Aind = get_global_id(0);
    int Bind = get_global_id(1);

	int Cind = Aind + Bind*_dim_size_X;

	if ( (Aind < _dim_size_X) && (Bind < _dim_size_star) )
	{ 

		float d = (X[Aind] - star[Bind]) * (X[Aind] - star[Bind]);

		float k = (sigma_f) * exp( (-1 / (2 * length_scale * length_scale)) * d);

		K_X_star[Cind] = k;

	}


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GAUSS PROCESS - Square Exponential Kernel + Noise
//
// K_X_star = (sigma_f_) * exp( ( -1 / (2 * l_scale_ * l_scale_) ) * ( (X - star)*(X - star) ) ) + (sigma_n_[0]);
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//K_X_star = (sigma_f_) * exp( ( -1 / (2 * l_scale_ * l_scale_) ) * ( (X - star)*(X - star) ) ) + (sigma_n_[0]);
kernel void Gauss_SquareExponentialKernelNoise (
    global const float * restrict X,
	int _dim_size_X,
	global const float * restrict star,
	int _dim_size_star,
	global float * restrict K_X_star,
	float sigma_f,
	float sigma_n,
	float length_scale
)

{ 
	int Aind = get_global_id(0);
    int Bind = get_global_id(1);

	int Cind = Aind + Bind*_dim_size_X;

	if ( (Aind < _dim_size_X) && (Bind < _dim_size_star) )
	{ 

		float d = (X[Aind] - star[Bind]) * (X[Aind] - star[Bind]);

		float k = (sigma_f) * exp( (-1 / (2 * length_scale * length_scale)) * d);

		if (Aind == Bind)
		{ 
			k = k + (sigma_n);
		}
		
		K_X_star[Cind] = k;


	}


}

//////////////////////////////////////
// EVAL - MATRIX VECTOR 
//////////////////////////////////////
kernel void Eval_matrixVector (
    global const float * restrict M, 
	int _rows,
    global const float * restrict V,
	int _cols,
    global float * restrict W
)

{

	int Aind = get_global_id(0);
    //int Bind = get_global_id(1);
    int Cind = Aind;

	if (Aind < _rows) 
	{ 
   
		float sum = 0;

		for (int i = 0; i < _cols; i++)
		{
			sum += M[Aind + i*_rows] * V[i];
		}

		W[Cind] = sum;

	}

}
 

/////////////////////////////////////////
// EVAL - MATRIX SUBTRACTION 
/////////////////////////////////////////
kernel void Eval_matrixSubtraction (
	global const float * restrict A,
    global const float * restrict B, 
    global float * restrict C,
    int ldc
)
{
  
    int Aind = get_global_id(0);
    int Bind = get_global_id(1);
    int Cind = Aind + Bind*ldc;

	C[Cind] = A[Cind] - B[Cind]; 

}

/////////////////////////////////////////////
// EVAL - MATRIX COPY
/////////////////////////////////////////////
kernel void Eval_matrixCopy (
    global const float * restrict A,
	global float * restrict C,
	int ldc
)
{
    
	int Aind = get_global_id(0);
    int Bind = get_global_id(1);
    int Cind = Aind + Bind*ldc;
	 
	C[Cind] = A[Cind];
	
}

/////////////////////////////////////////////
// EVAL - MATRIX INVERSE KERNELS
/////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//																																										  	
// BLAS - LEVEL 3																																						  
// xGEMM  
//
//	C := alpha*op(A)*op(B) + beta*C
//
//	op(X) = X, X^(T), X^(H)
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
kernel void matrixInverse_sGEMM (
	int TRANSA,
	int TRANSB,
	int M,
	int N,
	int K,
	int ALPHA,
	global float* A,
	int A_offset,
	int LDA,
	global float* B,
	int B_offset,
	int LDB,
	float BETA,
	global float* C,
	int C_offset,
	int LDC
)
{

    bool isANTrans = (TRANSA == 0);
    bool isBNTrans = (TRANSB == 0);

    if(isANTrans && isBNTrans)
    {
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                float value = 0.f;

                for(int l = 0; l < K; l++)
                {
                    value += MAT_ACCESS_col_major(A,i,l,LDA) * MAT_ACCESS_col_major(B,l,j,LDB);
					
                }

                MAT_ACCESS_col_major(C,i,j,LDC) = ALPHA * value + BETA * MAT_ACCESS_col_major(C,i,j,LDC);

            }
        }
    }

    if(isANTrans && !isBNTrans)
    {
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                float value = 0.f;

                for(int l = 0; l < K; l++)
                {
                    value += MAT_ACCESS_col_major(A,i,l,LDA) * MAT_ACCESS_col_major(B,j,l,LDB);
					
                }

                MAT_ACCESS_col_major(C,i,j,LDC) = ALPHA * value + BETA * MAT_ACCESS_col_major(C,i,j,LDC);

            }
        }
    }

    if(!isANTrans && isBNTrans)
    {
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                float value = 0.f;

                for(int l = 0; l < K; l++)
                {
                    value += MAT_ACCESS_row_major(A,i,l,LDA) * MAT_ACCESS_row_major(B,j,l,LDB);

                }

                MAT_ACCESS_col_major(C,i,j,LDC) = ALPHA * value + BETA * MAT_ACCESS_col_major(C,i,j,LDC);

            }
        }
    }

    if(!isANTrans && !isBNTrans)
    {
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                float value = 0.f;

                for(int l = 0; l < K; l++)
                {
                    
					value += MAT_ACCESS_row_major(A,i,l,LDA) * MAT_ACCESS_row_major(B,l,j,LDB);

				}

                MAT_ACCESS_col_major(C,i,j,LDC) = ALPHA * value + BETA * MAT_ACCESS_col_major(C,i,j,LDC);
            }
        }
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//																																										  	
// BLAS - LEVEL 3																																						  
// xTRSM 
//
//	B := alpha*op(A^(-1))*B
//
//	B := alpha*B*op(A^(-1))
//
//	op(A) = A, A^(T), A^(H)
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
kernel void matrixInverse_sTRSM (
	int SIDE,
	int UPLO,
	int TRANSA,
	int DIAG,
	int M,
	int N,
	float ALPHA,
	global float* A,
	int A_offset,
	int LDA,
	global float* B,
	int B_offset,
	int LDB
)
{

	//UPDATE WITH OFFSETS !!

    bool left = SIDE == 0;
    bool ltriangle = UPLO == 1;
    bool ntrans = TRANSA == 0;
    bool ndiag = DIAG == 0;

    if (left) 
	{
        if (ltriangle) 
		{
            if (ntrans) 
			{
                for (int i = 0; i < M; i++) 
				{
                    for (int j = 0; j < N; j++) 
					{
                        float this_x = MAT_ACCESS_col_major(B, i, j, LDB);
                        if (ndiag) this_x /= MAT_ACCESS_col_major(A, i, i, LDA);
                        for (int k = i+1; k < M; k++) 
						{
                            MAT_ACCESS_col_major(B, k, j, LDB) -= MAT_ACCESS_col_major(A, k, i, LDA)*this_x;
                        }
                        MAT_ACCESS_col_major(B, i, j, LDB) = this_x*ALPHA;
                    }
                }
            } 
			else 
			{ // ntrans
                for (int i = M-1; i >= 0; i--) 
				{
                    for (int j = 0; j < N; j++) 
					{
                        float this_x = MAT_ACCESS_col_major(B, i, j, LDB);
                        if (ndiag) this_x /= MAT_ACCESS_col_major(A, i, i, LDA);
                        for (int k = 0; k < i; k++) 
						{
                            MAT_ACCESS_col_major(B, k, j, LDB) -= MAT_ACCESS_col_major(A, i, k, LDA)*this_x;
                        }
                        MAT_ACCESS_col_major(B, i, j, LDB) = this_x*ALPHA;
                    }
                }
            }
        } 
		else 
		{ // ltriangle
            if (ntrans) 
			{
                for (int i = M-1; i >= 0; i--) 
				{
                    for (int j = 0; j < N; j++) 
					{
                        float this_x = MAT_ACCESS_col_major(B, i, j, LDB);
                        if (ndiag) this_x /= MAT_ACCESS_col_major(A, i, i, LDA);
                        for (int k = 0; k < i; k++) 
						{
                            MAT_ACCESS_col_major(B, k, j, LDB) -= MAT_ACCESS_col_major(A, k, i, LDA)*this_x;
                        }
                        MAT_ACCESS_col_major(B, i, j, LDB) = this_x*ALPHA;
                    }
                }
            } 
			else 
			{ // ntrans
                for (int i = 0; i < M; i++) 
				{
                    for (int j = 0; j < N; j++) 
					{
                        float this_x = MAT_ACCESS_col_major(B, i, j, LDB);
                        if (ndiag) this_x /= MAT_ACCESS_col_major(A, i, i, LDA);
                        for (int k = i+1; k < M; k++) 
						{
                            MAT_ACCESS_col_major(B, k, j, LDB) -= MAT_ACCESS_col_major(A, i, k, LDA)*this_x;
                        }
                        MAT_ACCESS_col_major(B, i, j, LDB) = this_x*ALPHA;
                    }
                }
            }
        }
    } 
	else 
	{ // left
        if (ltriangle) 
		{
            if (ntrans) 
			{
                for (int i = N-1; i >= 0; i--) 
				{
                    for (int j = 0; j < M; j++) 
					{
                        float this_x = MAT_ACCESS_col_major(B, j, i, LDB);
                        if (ndiag) this_x /= MAT_ACCESS_col_major(A, i, i, LDA);
                        for (int k = 0; k < i; k++) 
						{
                            MAT_ACCESS_col_major(B, j, k, LDB) -= MAT_ACCESS_col_major(A, i, k, LDA)*this_x;
                        }
                        MAT_ACCESS_col_major(B, j, i, LDB) = this_x*ALPHA;
                    }
                }
            } 
			else 
			{ // ntrans
                for (int i = 0; i < N; i++) 
				{
                    for (int j = 0; j < M; j++) 
					{
                        float this_x = MAT_ACCESS_col_major(B, j, i, LDB);
                        if (ndiag) this_x /= MAT_ACCESS_col_major(A, i, i, LDA);
                        for (int k = i+1; k < N; k++) 
						{
                            MAT_ACCESS_col_major(B, j, k, LDB) -= MAT_ACCESS_col_major(A, k, i, LDA)*this_x;
                        }
                        MAT_ACCESS_col_major(B, j, i, LDB) = this_x*ALPHA;
                    }
                }
            }
        } 
		else 
		{ // ltriangle
            if (ntrans) 
			{
                for (int i = 0; i < N; i++) 
				{
                    for (int j = 0; j < M; j++) 
					{
                        float this_x = MAT_ACCESS_col_major(B, j, i, LDB);
                        if (ndiag) this_x /= MAT_ACCESS_col_major(A, i, i, LDA);
                        for (int k = i+1; k < N; k++) 
						{
                            MAT_ACCESS_col_major(B, j, k, LDB) -= MAT_ACCESS_col_major(A, i, k, LDA)*this_x;
                        }
                        MAT_ACCESS_col_major(B, j, i, LDB) = this_x*ALPHA;
                    }
                }
            } 
			else 
			{ // ntrans
                for (int i = N-1; i >= 0; i--) 
				{
                    for (int j = 0; j < M; j++) 
					{
                        float this_x = MAT_ACCESS_col_major(B, j, i, LDB);
                        if (ndiag) this_x /= MAT_ACCESS_col_major(A, i, i, LDA);
                        for (int k = 0; k < i; k++) 
						{
                            MAT_ACCESS_col_major(B, j, k, LDB) -= MAT_ACCESS_col_major(A, k, i, LDA)*this_x;
                        }
                        MAT_ACCESS_col_major(B, j, i, LDB) = this_x*ALPHA;
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//																																										  	
// BLAS - LEVEL 3																																						  
// xSYRK  
//
//	C := alpha*A*A^(T) + beta*C
//
//	C := alpha*A^(T)*A + beta*C
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
kernel void matrixInverse_sSYRK (
	int UPLO,
	int TRANS,
	int N,
	int K,
	float ALPHA,
	global float* A,
	int A_offset,
	int LDA,
	float BETA,
	global float* C,
	int C_offset,
	int LDC
)
{

    bool ltriangle = UPLO == 1;
    bool ntrans = TRANS == 0;

    int ind_m = get_global_id(0);
    int ind_n = get_global_id(1);

    bool upper = ind_m < ind_n;
    bool lower = ind_m > ind_n;

    // Early return for work items in non-referenced parts
    if (ltriangle && upper) return;
    if (!ltriangle && lower) return;

    float value = 0.f;

    if (ntrans) 
	{
            for (int i = 0; i < K; i++) 
			{
                value += MAT_ACCESS_col_major(A, ind_m, i, LDA)*MAT_ACCESS_col_major(A, ind_n, i, LDA);
            }
    } 
	else 
	{
            for (int i = 0; i < K; i++) 
			{
                value += MAT_ACCESS_col_major(A, i, ind_m, LDA)*MAT_ACCESS_col_major(A, i, ind_n, LDA);
            }
    }

    value *= ALPHA;
    MAT_ACCESS_col_major(C, ind_m, ind_n, LDC) = BETA*MAT_ACCESS_col_major(C, ind_m, ind_n, LDC) + value;

}
