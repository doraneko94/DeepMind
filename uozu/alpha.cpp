#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
using namespace std;

const int input_height = 3;
const int input_width = 3;
const int input_channels = 7;

double relu(double x) {
    if (x < 0) return 0;
    else return x;
}

const int n_hidden_in = 2 * 2 * 8;
const int n_hidden = 16;

double sigmoid(double x) {
    return 1.0 / (1 + exp(-x));
}

double tanh(double x) {
    return (exp(x)-exp(-x)) / (exp(x)+exp(-x));
}

const int n_outputs1 = 9;
// const int n_outputs2 = 1;
double prob[n_outputs1] = {0};

double filter1[4][2][2] = {{{1.0, 1.0},
                            {0.0, 1.0}},
                           {{1.0, 0.0},
                            {0.0, 1.0}},
                           {{1.0, 0.0},
                            {0.0, 0.0}},
                           {{0.0, 1.0},
                            {1.0, 0.0}}};
double filter2[8][1][1] = {{{1.0}}, {{0.1}}, {{0.2}}, {{0.3}},
                           {{0.4}}, {{0.5}}, {{0.6}}, {{0.7}}};

double W_hidden1[n_hidden_in][n_hidden];
// double W_hidden2[n_hidden_in][n_hidden];
double W_hidden3[n_hidden][n_hidden];
double W_p[n_hidden][n_outputs1];
// double W_v[n_hidden][n_outputs2];
double b_hidden1[n_hidden] = {0};
double b_hidden3[n_hidden] = {0};
double b_p[n_outputs1] = {0};

void W_init() {
    for (int i=0; i<n_hidden_in; i++) for (int j=0; j<n_hidden; j++) W_hidden1[i][j] = 0.3;
    for (int i=0; i<n_hidden; i++) for (int j=0; j<n_hidden; j++) W_hidden3[i][j] = 0.5;
    for (int i=0; i<n_hidden; i++) for (int j=0; j<n_outputs1; j++) W_p[i][j] = 0.4;
    for (int i=0; i<n_hidden; i++) b_hidden1[i] = 0.5;
    for (int i=0; i<n_hidden; i++) b_hidden3[i] = 1.0;
    for (int i=0; i<n_outputs1; i++) b_p[i] = 0.1;
}

int GameBoard[input_height][input_width] = {0};
int InputData[input_height][input_width][input_channels] = {0};
int obs1[input_height][input_width] = {0};
int obs2[input_height][input_width] = {0};
int obs3[input_height][input_width] = {0};
double conv_out1[4][2][2] ={0};
double conv_out2[8][2][2] ={0};
double hidden_in[n_hidden_in] = {0};
double hidden1_out[n_hidden] = {0};
// double hidden2_out[n_hidden] = {0};
double hidden3_out[n_hidden] = {0};
double p_out[n_outputs1] = {0};
// double v_out[n_outputs2] = {0};
int arg_index[n_outputs1] = {0};
int player = 1;
int playerA, playerB;

void MakeInputData() {
    for (int h=0; h<input_height; h++) for (int w=0; w<input_width; w++) {
        if (GameBoard[h][w]==1) InputData[h][w][0] = 1;
        else InputData[h][w][0] = 0;
    }
    for (int h=0; h<input_height; h++) for (int w=0; w<input_width; w++) {
        if (GameBoard[h][w]==-1) InputData[h][w][1] = -1;
        else InputData[h][w][1] = 0;
    }
    for (int h=0; h<input_height; h++) for (int w=0; w<input_width; w++) {
        if (obs1[h][w]==1) InputData[h][w][2] = 1;
        else InputData[h][w][2] = 0;
    }
    for (int h=0; h<input_height; h++) for (int w=0; w<input_width; w++) {
        if (obs1[h][w]==-1) InputData[h][w][3] = -1;
        else InputData[h][w][3] = 0;
    }
    for (int h=0; h<input_height; h++) for (int w=0; w<input_width; w++) {
        if (obs3[h][w]==1) InputData[h][w][4] = 1;
        else InputData[h][w][4] = 0;
    }
    for (int h=0; h<input_height; h++) for (int w=0; w<input_width; w++) {
        if (obs3[h][w]==-1) InputData[h][w][5] = -1;
        else InputData[h][w][5] = 0;
    }
    for (int h=0; h<input_height; h++) for (int w=0; w<input_width; w++) {
        InputData[h][w][6] = player;
    }
}

void ShowBoard() {
    for (int h=0; h<input_height; h++) {
        for (int w=0; w<input_width; w++) {
            if (GameBoard[h][w]==1) cout << " x";
            if (GameBoard[h][w]==-1) cout << " o";
            if (GameBoard[h][w]==0) cout << " .";
        }
        cout << endl;
    }
    if (player==playerA) cout << "player: A" << endl;
    else cout << "player: B" << endl;
    cout << " " << endl;
}

bool isValid(int a) {
    int row = a / 3;
    int col = a - (row * 3);
    if (0 <= a && a <= 8 && GameBoard[row][col]==0) return true;
    else return false;
}

void move(int a) {
    int row = a / 3;
    int col = a - (row * 3);
    for (int h=0; h<input_height; h++) for (int w=0; w<input_width; w++) {
        (*obs3)[h*input_height+w] = (*obs2)[h*input_height+w];
        (*obs2)[h*input_height+w] = (*obs1)[h*input_height+w];
        (*obs1)[h*input_height+w] = (*GameBoard)[h*input_height+w];    
    }
    GameBoard[row][col] = player;
    player *= -1;
}

int isDone() {
    for (int h=0; h<input_height; h++) {
        int sum = 0;
        for (int w=0; w<input_width; w++) sum += GameBoard[h][w];
        if (sum==3) return 1;
        if (sum==-3) return -1;
    }
    for (int w=0; w<input_width; w++) {
        int sum = 0;
        for (int h=0; h<input_height; h++) sum += GameBoard[h][w];
        if (sum==3) return 1;
        if (sum==-3) return -1;
    }
    int sum1 = GameBoard[0][0] + GameBoard[1][1] + GameBoard[2][2];
    if (sum1==3) return 1;
    if (sum1==-3) return -1;
    int sum2 = GameBoard[0][2] + GameBoard[1][1] + GameBoard[2][0];
    if (sum2==3) return 1;
    if (sum2==-3) return -1;
    int zeros = 0;
    for (int h=0; h<input_height; h++) for (int w=0; w<input_width; w++) {
        if (GameBoard[h][w]==0) zeros++;
    }
    if (zeros==0) return 0;
    return -2;
}

void cointoss() {
    playerA = 1;
    playerB = -playerA;
}

void conv1() {
    for (int f=0; f<4; f++) {
        for (int hs=0; hs<2; hs++) for (int ws=0; ws<2; ws++) {
            double temp = 0;
            for (int c=0; c<7; c++) {
                for (int hf=0; hf<2; hf++) for (int wf=0; wf<2; wf++) {
                    temp += filter1[f][hf][wf] * InputData[hs+hf][ws+wf][c];
                }
            }
            conv_out1[f][hs][ws] = temp;
        }
    }
}

void conv2() {
    for (int f=0; f<8; f++) {
        for (int hs=0; hs<2; hs++) for (int ws=0; ws<2; ws++) {
            double temp = 0;
            for (int c=0; c<4; c++) {
                for (int hf=0; hf<1; hf++) for (int wf=0; wf<1; wf++) {
                    temp += filter1[f][hf][wf] * InputData[hs+hf][ws+wf][c];
                }
            }
            conv_out2[f][hs][ws] = temp;
        }
    }
}

void flatten() {
    int index = 0;
    for (int f=0; f<8; f++) for (int h=0; h<2; h++) for (int w=0; w<2; w++) {
        hidden_in[index] = conv_out2[f][h][w];
        index++;
    }
}

void hidden1() {
    for (int i=0; i<n_hidden; i++) {
        int temp = 0;
        for (int j=0; j<n_hidden_in; j++) temp += W_hidden1[j][i] * hidden_in[j];
        hidden1_out[i] = temp + b_hidden1[i];
    }
}
/* 
void hidden2() {
    for (int i=0; i<n_hidden; i++) {
        int temp = 0;
        for (int j=0; j<n_hidden_in; j++) temp += W_hidden2[j][i] * hidden_in[j];
        hidden2_out[i] = temp;
    }
}
*/
void hidden3() {
    for (int i=0; i<n_hidden; i++) {
        int temp = 0;
        for (int j=0; j<n_hidden; j++) temp += W_hidden3[j][i] * hidden1_out[j];
        hidden3_out[i] = temp + b_hidden3[i];
    }
}

void p_layer() {
    for (int i=0; i<n_outputs1; i++) {
        int temp = 0;
        for (int j=0; j<n_hidden; j++) temp += W_p[j][i] * hidden3_out[j];
        p_out[i] = temp + b_p[i];
    }
}

void softmax() {
    double denom = 0.0;
    for (int i=0; i<n_outputs1; i++) denom += exp(p_out[i]);
    for (int i=0; i<n_outputs1; i++) prob[i] = exp(p_out[i]) / denom;
}

bool key(int x, int y) {
    return prob[x] > prob[y];
}

void argsort() {
    for (int i=0; i<n_outputs1; i++) arg_index[i] = i;
    sort(arg_index, arg_index+n_outputs1, key);
}

void CNN() {
    // cout << "A" << endl;
    MakeInputData();
    // cout << "B" << endl;
    conv1();
    // cout << "C" << endl;
    conv2();
    // cout << "D" << endl;
    flatten();
    // cout << "E" << endl;
    hidden1();
    // cout << "F" << endl;
    hidden3();
    // cout << "G" << endl;
    p_layer();
    // cout << "H" << endl;
    softmax();
    // cout << "I" << endl;
    argsort();
}

int main() {
    cointoss();
    W_init();
    while (isDone() < -1){
        ShowBoard();
        int a = -1;
        if (player==playerA) {
            while (!isValid(a)) cin >> a;
        } else {
            CNN();
            int index = 0;
            while (!isValid(a)) {
                a = arg_index[index];
                index++;
            }
        }
        move(a);
    }
    int result = isDone();
    ShowBoard();
    if (result == playerA) cout << "win!" << endl;
    else if (result == 0) cout << "draw" << endl;
    else cout << "lose..." << endl;
    return 0; 
}