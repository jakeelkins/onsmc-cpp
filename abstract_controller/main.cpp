#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <string>
#include <fstream>
#include <utility> // std::pair

#include "onsmc.h"

using namespace std;

void print_matrix(vector<float>& Z, unsigned int rows, unsigned int cols);
void dynamics(float* y_ddot, float* u, float* y, float* y_dot, int output_dim);
void desired_trajectory(float* yd, float* yd_dot, float* yd_ddot, float t);
void write_csv(string filename, vector<std::pair<string, vector<float>>> dataset);

int main(){

    // ---- hypers ----
    int input_dim = 3;
    int output_dim = 1;

    float dt = 0.001;
    float tf = 5.0;

    // controller
    ONSMC controller(input_dim, output_dim, dt);

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

    // variables needed
    vector<float> u (
        output_dim,
        0.0f
    );


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

    // -------
    
    unsigned int save_idx = 0;
    printf("beginning loop... \n");
    for (float t = 0.0f; t<tf; t+=dt){

        // get desired trajectory
        printf("getting desired traj... \n");
        desired_trajectory(yd.data(), yd_dot.data(), yd_ddot.data(), t);

        // get control
        printf("getting control... \n");
        controller.get_control(u.data(), y.data(), y_dot.data(),
                                yd.data(), yd_dot.data(), yd_ddot.data());

        printf("t: %f \t e[0]: %f \n", t, controller.e[0]);
        // --- save all ---
        t_save[save_idx] = t;
        y_save[save_idx] = y[0];
        yd_save[save_idx] = yd[0];
        s_save[save_idx] = controller.s[0];

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
