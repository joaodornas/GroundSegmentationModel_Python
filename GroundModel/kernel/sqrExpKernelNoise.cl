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