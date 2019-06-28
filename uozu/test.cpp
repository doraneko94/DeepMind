#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
using namespace std;

int arg_index[5] = {0};
double prob[5] = {0.1, 0.5, 0.2, 0.3, 0.4};

bool key(int x, int y) {
    return prob[x] > prob[y];
}

void argsort() {
    for (int i=0; i<5; i++) arg_index[i] = i;
    sort(arg_index, arg_index+5, key);
}

int main() {
    argsort();
    cout << arg_index[0] << arg_index[1] << arg_index[2] << arg_index[3] << arg_index[4] << endl;
    return 0;
}