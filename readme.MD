機械学習のうち言語処理を利用したリポジトリです

// test.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include<iostream>
#include<fstream>
#include<string>
#include<sstream> 
#include<vector>

using namespace std;

vector<float> split(const string &s, char delim) {
	vector<float> elems;
	stringstream ss(s);
	string item;
	while (getline(ss, item, delim)) {
		if (!item.empty()) {
			elems.push_back( atof(item.c_str()) );
		}
	}
	return elems;
}


vector<vector<float>> read_csv(string fileName) {
	ifstream ifs(fileName);
	string str;

	vector<vector<float>> ret;
	
	while (getline(ifs, str))
	{
		ret.push_back(split(str.c_str(),','));
	}
	return ret;
}

vector<float> cos_sim(vector<float> y, vector<float> label) {

}


int main()
{
// 

	vector<vector<float>> data = read_csv("test.csv");
	int i = 0;
	int j = 0;
	cout << data.size() <<endl;
	cout << data[0].size() << endl;
	for (i = 0; i < data.size(); i++) {
		for (j = 0; j < data.at(i).size(); j++) 
			cout << data.at(i).at(j);
		cout << endl;
	}//*/


	cout << endl;
    return 0;
}

