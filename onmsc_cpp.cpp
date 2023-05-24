#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <string>
#include <fstream>
#include <utility> // std::pair

using namespace std;

class NeuralNetwork {

    public:

        int input_dim;
        int hidden_dim;
        int output_dim;

        // network params
        vector<float> W;
        vector<float> V;
        vector<float> W_T;
        vector<float> V_T;

        // calcs used for backprop
        vector<float> sigma_x;
        vector<float> y_hat;

        // constructor
        NeuralNetwork(unsigned int _input_dim, unsigned int _hidden_dim, unsigned int _output_dim){

            input_dim = _input_dim;
            hidden_dim = _hidden_dim;
            output_dim = _output_dim;

            // first vector of weights
            vector<float> _V (
                input_dim*hidden_dim,
                0.0f
            );

            // second vector of weights
            vector<float> _W (
                hidden_dim*output_dim,
                0.0f
            );

            // transposes
            vector<float> _V_T (
                input_dim*hidden_dim,
                0.0f
            );

            vector<float> _W_T (
                hidden_dim*output_dim,
                0.0f
            );

            V = _V;
            W = _W;

            V_T = _V_T;
            W_T = _W_T;

            // -- call internal random_init method --

            mt19937 rng(5); // I put another RNG inbstance here just because I don't want to pass it to NN

            _random_init(rng, V.data(), input_dim, hidden_dim);
            _random_init(rng, W.data(), hidden_dim, output_dim);

        }

        void forward(float* x){
            // forward pass.
            // x is the input.
            // access the output outside the class by NN.y_hat.

            // -- init the interim calculations --
            vector<float> V_T_x (
                hidden_dim,
                0.0f
            );

            vector<float> _sigma_x (
                hidden_dim,
                0.0f
            );

            vector<float> _y_hat (
                output_dim,
                0.0f
            );

            sigma_x = _sigma_x;
            y_hat = _y_hat;

            // run the forward pass with our reference gemv: (note: x is already a pointer)
            // V^T * x
            _transpose(V_T.data(), V.data(), input_dim, hidden_dim);
            _gemv(V_T_x.data(), V_T.data(), x, hidden_dim, input_dim);

            // sigma_x = sigmoid(V_T_x)
            _sigmoid(sigma_x.data(), V_T_x.data(), hidden_dim);

            // append another 1 to the activation
            sigma_x[0] = 1.0f;

            // W^T * sigmoid(V^T x)
            _transpose(W_T.data(), W.data(), hidden_dim, output_dim);
            _gemv(y_hat.data(), W_T.data(), sigma_x.data(), output_dim, hidden_dim);

        }

        void update_W(float* W_dot, float dt){
            // W = W + dW
            // W is hidden_dim x output_dim
            for (unsigned int i = 0; i < (hidden_dim*output_dim); ++i){
                W[i] = W[i] + W_dot[i]*dt;
            }
        }

        void update_V(float* V_dot, float dt){
            // V = V + dV
            // V is input_dim x hidden_dim
            for (unsigned int i = 0; i < (input_dim*hidden_dim); ++i){
                V[i] = V[i] + V_dot[i]*dt;
            }
        }

        // --internal methods--
        void _random_init(mt19937 &rng, float* W, int N, int M);
        void _gemv(float* y, float* A, float* x, int N, int M);
        void _transpose(float* A_T, float* A, int N, int M);
        void _sigmoid(float* y, float* x, int N);
        void print_matrix(vector<float>& Z, unsigned int rows, unsigned int cols);
};

void print_matrix(vector<float>& Z, unsigned int rows, unsigned int cols);
void dynamics(float* y_ddot, float* u, float* y, float* y_dot, int output_dim);
void desired_trajectory(float* yd, float* yd_dot, float* yd_ddot, float t);
void sat(float* s_out, float* s_in, float phi, int N);
void gemv(float* y, float* A, float* x, int N, int M);
void gemm(float* C, float* A, float* B, int N, int P, int M);
void transpose(float* A_T, float* A, int N, int M);
void sigmoid_prime(float* sigma_x_prime, float* sigma_x, int hidden_dim);
void elementwise_subtraction(float* y, float* a, float* b, int N);
void elementwise_addition(float* y, float* a, float* b, int N);
void elementwise_multiplication(float* y, float* a, float* b, int N);
void write_csv(string filename, vector<std::pair<string, vector<float>>> dataset);

int main(){

    // ---- hypers ----
    int input_dim = 3;
    int hidden_dim = 10;
    int output_dim = 1;

    float dt = 0.001;
    float tf = 5.0;

    float eta = 2.0;
    float Lambda_val = 4.0;
    float D = 0.5;
    float phi = 0.0005;
    float H_val = 0.95;

    float F_val = 50.0;
    float G_val = 30.0;

    // ---- ICs ----
    vector<float> y (
        output_dim,
        0.0f
    );

    vector<float> y_dot (
        output_dim,
        0.0f
    );

    vector<float> y_ddot (
        output_dim,
        0.0f
    );

    vector<float> yd (
        output_dim,
        0.0f
    );

    vector<float> yd_dot (
        output_dim,
        0.0f
    );

    vector<float> yd_ddot (
        output_dim,
        0.0f
    );

    vector<float> M_hat (
        output_dim,
        1.0f
    );

    // variables needed
    vector<float> x (
        input_dim,
        0.0f
    );

    vector<float> u (
        output_dim,
        0.0f
    );

    vector<float> sigma_x_prime (
        hidden_dim,
        0.0f
    );

    vector<float> sigma_x_prime_matrix (
        hidden_dim*hidden_dim,
        0.0f
    );

    vector<float> V_dot (
        input_dim*hidden_dim,
        0.0f
    );

    vector<float> s (
        output_dim,
        0.0f
    );

    vector<float> e (
        output_dim,
        0.0f
    );

    vector<float> e_dot (
        output_dim,
        0.0f
    );

    vector<float> sat_s (
        output_dim,
        0.0f
    );

    vector<float> y_ddot_r (
        output_dim,
        0.0f
    );

    vector<float> s_delta (
        output_dim,
        0.0f
    );

    vector<float> s_delta_T (
        output_dim,
        0.0f
    );

    vector<float> Lambda_e (
        output_dim,
        0.0f
    );

    vector<float> Lambda_e_dot (
        output_dim,
        0.0f
    );

    // -- interim update calcs --
    vector<float> sigma_x_s_delta (
        output_dim,
        0.0f
    );

    vector<float> W_T_sigma_x_prime (
        output_dim*hidden_dim,
        0.0f
    );

    vector<float> G_x_s_delta_T (
        input_dim*output_dim,
        0.0f
    );

    vector<float> W_dot (
        hidden_dim*output_dim,
        0.0f
    );

    vector<float> M_hat_dot (
        output_dim,
        0.0f
    );

    // --- populate F, G, H and Lambda ---
    vector<float> F (
        hidden_dim*hidden_dim,
        0.0f
    );

    for (unsigned int i = 0; i<(hidden_dim); ++i){
        for (unsigned int j = 0; j<(hidden_dim); ++j){
            if (i==j){
                F[hidden_dim*i + j] = F_val;
            }
        }
    }

    vector<float> G (
        input_dim*input_dim,
        0.0f
    );

    for (unsigned int i = 0; i<(input_dim); ++i){
        for (unsigned int j = 0; j<(input_dim); ++j){
            if (i==j){
                G[input_dim*i + j] = G_val;
            }
        }
    }

    vector<float> H (
        output_dim,
        H_val
    );

    vector<float> Lambda (
        output_dim*output_dim,
        0.0f
    );

    for (unsigned int i = 0; i<(output_dim); ++i){
        for (unsigned int j = 0; j<(output_dim); ++j){
            if (i==j){
                Lambda[output_dim*i + j] = Lambda_val;
            }
        }
    }

    // ---- for saving ----
    vector<float> t_save (
        (int)(tf/dt),
        0.0f
    );

    vector<float> y_save (
        (int)(tf/dt),
        0.0f
    );

    vector<float> yd_save (
        (int)(tf/dt),
        0.0f
    );

    vector<float> s_save (
        (int)(tf/dt),
        0.0f
    );


    // build NN:
    NeuralNetwork NN(input_dim, hidden_dim, output_dim);

    // -------
    unsigned int save_idx = 0;
    for (float t = 0.0f; t<tf; t+=dt){
        // get desired trajectory
        //printf("getting desired trajectory... \n");
        desired_trajectory(yd.data(), yd_dot.data(), yd_ddot.data(), t);

        // e = yd - y
        elementwise_subtraction(e.data(), yd.data(), y.data(), output_dim);
        // e_dot = yd_dot - y_dot;
        elementwise_subtraction(e_dot.data(), yd_dot.data(), y_dot.data(), output_dim);
        // Lambda_e = Lambda*e
        gemv(Lambda_e.data(), Lambda.data(), e.data(), output_dim, output_dim);
        // s = e_dot + Lambda*e;
        elementwise_addition(s.data(), e_dot.data(), Lambda_e.data(), output_dim);
        // Lambda_e_dot = Lambda*e_dot
        gemv(Lambda_e_dot.data(), Lambda.data(), e_dot.data(), output_dim, output_dim);
        // y_ddot_r = yd_ddot + Lambda*e_dot;
        elementwise_addition(y_ddot_r.data(), yd_ddot.data(), Lambda_e_dot.data(), output_dim);

        // sat(s)
        //printf("sat(s)... \n");
        sat(sat_s.data(), s.data(), phi, output_dim);

        // make state vector
        x[0] = y[0];
        x[1] = y_dot[0];
        x[2] = 1.0f; // appended 1

        //get f_x
        //printf("forward pass... \n");
        NN.forward(x.data());

        // control law is u = M_hat*y_ddot_r + NN.y_hat + M_hat*(D + eta)*sat_s;
        // i.e. u = M_hat*(y_ddot_r + (D + eta)*sat_s) + NN.y_hat
        // u <- (D + eta)*sat_s
        for (unsigned int i = 0; i<output_dim; ++i){
            u[i] = (D + eta)*sat_s[i];
        }
        // now add y_ddot_r with (D + eta)*sat_s
        // u <- y_ddot_r + (D + eta)*sat_s, so u <- y_ddot_r + u
        elementwise_addition(u.data(), y_ddot_r.data(), u.data(), output_dim);
        // u <- M_hat*(y_ddot_r + (D + eta)*sat_s), so u <- M_hat*u
        //gemv(u.data(), M_hat.data(), u.data(), output_dim, output_dim);  // non diagonal M case
        elementwise_multiplication(u.data(), M_hat.data(), u.data(), output_dim); // diagonal M case
        // u <- M_hat*(y_ddot_r + (D + eta)*sat_s) + NN.y_hat, so u <- u + NN.y_hat
        elementwise_addition(u.data(), u.data(), NN.y_hat.data(), output_dim);


        // printf("V: \n");
        // print_matrix(NN.V, input_dim, hidden_dim);
        // printf("W: \n");
        // print_matrix(NN.W, hidden_dim, output_dim);

        printf("t: %f \t e[0]: %f \n", t, e[0]);

        // --- updates ---
        // get sigma prime
        sigmoid_prime(sigma_x_prime.data(), NN.sigma_x.data(), hidden_dim);

        // populate sigma_x_prime_matrix with diagonal
        for (unsigned int i = 0; i<hidden_dim; ++i){
            for (unsigned int j = 0; j<hidden_dim; ++j){
                if (i == j){
                    sigma_x_prime_matrix[i*hidden_dim + j] = sigma_x_prime[i];
                }
            }
        }

        //s_delta = s - phi*sat_s;
        for (unsigned int i = 0; i<output_dim; ++i){
            s_delta[i] = s[i] - phi*sat_s[i];
        }

        // W
        // sigma_x*s_delta^T
        //printf("updating... \n");
        transpose(s_delta_T.data(), s_delta.data(), hidden_dim, 1);
        gemm(sigma_x_s_delta.data(), NN.sigma_x.data(), s_delta_T.data(), hidden_dim, 1, output_dim);
        // F * sigma_x*s_delta^T
        gemm(W_dot.data(), F.data(), sigma_x_s_delta.data(), hidden_dim, hidden_dim, output_dim);
        NN.update_W(W_dot.data(), dt);

        // V
        // W^T sigma_x_prime
        gemm(W_T_sigma_x_prime.data(), NN.W_T.data(), sigma_x_prime_matrix.data(), output_dim, hidden_dim, hidden_dim);
        // x s_delta^T ( use G_x_s_delta because same size)
        gemm(G_x_s_delta_T.data(), x.data(), s_delta_T.data(), input_dim, 1, output_dim);
        gemm(G_x_s_delta_T.data(), G.data(), G_x_s_delta_T.data(), input_dim, input_dim, output_dim);
        gemm(V_dot.data(), G_x_s_delta_T.data(), W_T_sigma_x_prime.data(), input_dim, output_dim, hidden_dim);
        NN.update_V(V_dot.data(), dt);

        // H (do diagonal version)
        for (unsigned int i = 0; i<output_dim; ++i){
            M_hat_dot[i] = H[i]*(y_ddot_r[i] + (D + eta)*sat_s[i])*s_delta[i];
            // update
            M_hat[i] += M_hat_dot[i]*dt;
        }

        // --- save all ---
        t_save[save_idx] = t;
        y_save[save_idx] = y[0];
        yd_save[save_idx] = yd[0];
        s_save[save_idx] = s[0];

        // integrate dynamics
        //printf("integrating... \n");
        dynamics(y_ddot.data(), u.data(), y.data(), y_dot.data(), output_dim);
        for (unsigned int i = 0; i<output_dim; ++i){
            y_dot[i] += y_ddot[i]*dt;
            y[i] += y_dot[i]*dt;
        }

        ++save_idx;
    }

    // wrap and save
    vector<std::pair<string, vector<float>>> out_dataset = {{"t", t_save}, {"y", y_save}, {"yd", yd_save}, {"s", s_save}};
    
    // Write the vector to CSV
    write_csv("sim_out.csv", out_dataset);
    
    return 0;
}


void print_matrix(vector<float>& Z, unsigned int rows, unsigned int cols){
    for (unsigned int row = 0; row < rows; ++row){
        for (unsigned int col = 0; col < cols; ++col){
            printf("%.4f ",Z[row*cols + col]);
        }
        printf("\n");
    }
}

void dynamics(float* y_ddot, float* u, float* y, float* y_dot, int output_dim){
    float h = 1.0f;
    float a_1 = 1.0f;
    float a_2 = 1.0f;

    // would loop here if ouput_dim > 0.

    y_ddot[0] = (u[0] - a_1*(y[0]*y[0]) - a_2*y_dot[0])/h;
}

void desired_trajectory(float* yd, float* yd_dot, float* yd_ddot, float t){
    float y1d = sin(2*t);
    float y1d_dot = 2*cos(2*t);
    float y1d_ddot = -4*sin(2*t);

    yd[0] = y1d;
    yd_dot[0] = y1d_dot;
    yd_ddot[0] = y1d_ddot;
}

void sat(float* s_out, float* s_in, float phi, int N){
    // vectorized sat function for SMC
    for (unsigned int i = 0; i<N; ++i){
        float s_in_i = s_in[i];
        if (s_in_i < -phi){
            s_out[i] = -1.0f;
        } else if (s_in_i > phi){
            s_out[i] = 1.0f;
        } else {
            s_out[i] = s_in_i/phi;
        }
    }
}

// void sat(float* s_out, float s_in, float phi){
//     // scalar sat function for SMC
//     if (s_in < -phi){
//         *s_out = -1.0f;
//     } else if (s_in > phi){
//         *s_out = 1.0f;
//     } else {
//         *s_out = s_in/phi;
//     }
// }

void gemv(float* y, float* A, float* x, int N, int M){
    // y = Ax, where A is NxM.
    for (unsigned int row = 0; row < N; ++row){
        float sum_ = 0.0f;
        for (unsigned int col = 0; col < M; ++col){
            sum_ += A[row*M + col]*x[col];
        }
        y[row] = sum_;
    }
}

void gemm(float* C, float* A, float* B, int N, int P, int M){
    // C = AB, where C is NxM, A NxP, B PxM.
    for (unsigned int i = 0; i < N; ++i){
        for (unsigned int j = 0; j < M; ++j){
            float sum_ = 0.0f;
            for (unsigned int k = 0; k < P; ++k){
                sum_ += A[i*P + k]*B[k*M + j];
            }
            C[i*M + j] = sum_;
        }
    }
}


void transpose(float* A_T, float* A, int N, int M){
    // single matrix transpose: A is NxM, A_T is MxN.
    for (unsigned int i = 0; i<N; ++i){
        for (unsigned int j = 0; j<M; ++j){
            A_T[j*N + i] = A[i*M + j];
        }
    }
}

void sigmoid_prime(float* sigma_x_prime, float* sigma_x, int hidden_dim){
    // makes vector of sigmoid prime
    for (unsigned int i = 0; i<hidden_dim; ++i){
        float sgm_x = sigma_x[i];
        sigma_x_prime[i] = sgm_x*(1 - sgm_x);
    }
}

void elementwise_subtraction(float* y, float* a, float* b, int N){
    // y = a - b, all Nx1.
    for (unsigned int i = 0; i<N; ++i){
        y[i] = a[i] - b[i];
    }
}

void elementwise_addition(float* y, float* a, float* b, int N){
    // y = a + b, all Nx1.
    for (unsigned int i = 0; i<N; ++i){
        y[i] = a[i] + b[i];
    }
}

void elementwise_multiplication(float* y, float* a, float* b, int N){
    // y = a * b, all Nx1.
    for (unsigned int i = 0; i<N; ++i){
        y[i] = a[i]*b[i];
    }
}


void write_csv(string filename, vector<std::pair<string, vector<float>>> dataset){
    // Make a CSV file with one or more columns of float values
    // Each column of data is represented by the pair <column name, column data>
    //   as std::pair<std::string, std::vector<float>>
    // The dataset is represented as a vector of these columns
    // Note that all columns should be the same size
    
    // Create an output filestream object
    ofstream myFile(filename);
    
    // Send column names to the stream
    for(int j = 0; j < dataset.size(); ++j)
    {
        myFile << dataset.at(j).first;
        if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
    }
    myFile << "\n";
    
    // Send data to the stream
    for(int i = 0; i < dataset.at(0).second.size(); ++i)
    {
        for(int j = 0; j < dataset.size(); ++j)
        {
            myFile << dataset.at(j).second.at(i);
            if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
        }
        myFile << "\n";
    }
    
    // Close the file
    myFile.close();
}


void NeuralNetwork::_random_init(mt19937 &rng, float* W, int N, int M){
    // samples random values for weights and biases.
    // generalized for a matrix NxM.
    // init like pytorch does, uniform ()
    //float stddev = 1/sqrt((float)N);
    float stddev = 0.01;
    uniform_real_distribution<float> uni(-stddev, stddev);

    for (unsigned int i = 0; i<N; ++i){
        for (unsigned int j = 0; j<M; ++j){
            auto rand_val = uni(rng);
            W[i*M + j] = rand_val;
        }
    }
}

void NeuralNetwork::_gemv(float* y, float* A, float* x, int N, int M){
    // y = Ax, where A is NxM.
    for (unsigned int row = 0; row < N; ++row){
        float sum_ = 0.0f;
        for (unsigned int col = 0; col < M; ++col){
            sum_ += A[row*M + col]*x[col];
        }
        y[row] = sum_;
    }
}

void NeuralNetwork::_transpose(float* A_T, float* A, int N, int M){
    // single matrix transpose: A is NxM, A_T is MxN.
    for (unsigned int i = 0; i<N; ++i){
        for (unsigned int j = 0; j<M; ++j){
            A_T[j*N + i] = A[i*M + j];
        }
    }
}

void NeuralNetwork::_sigmoid(float* y, float* x, int N){
    // elementwise sigmoid activation
    // input x, output y, N is length of x.
    for (unsigned int i = 0; i<N; ++i){
        y[i] = 1/(1 + exp(-x[i]));
    }
}

void NeuralNetwork::print_matrix(vector<float>& Z, unsigned int rows, unsigned int cols){
    for (unsigned int row = 0; row < rows; ++row){
        for (unsigned int col = 0; col < cols; ++col){
            printf("%.4f ",Z[row*cols + col]);
        }
        printf("\n");
    }
}
