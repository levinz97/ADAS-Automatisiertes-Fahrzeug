#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

int smaisn()
{
    string classNames = "./Mydnn/coco.names";
    ifstream infile( classNames.c_str(), ios::binary );
    ofstream outfile( "./Mydnn/classNames.txt", ios::binary | ios::trunc );
    // cout << outfile.is_open() << endl;
    if( !infile.is_open() )
    {
        cout << "failed to open the file !" << endl;
        return -1;
    }
    string fileLine;
    while( getline( infile, fileLine ) && !fileLine.empty() )
    {
        cout << fileLine << endl;
        istringstream inFileLine( fileLine );
        string temp;
        size_t cnt = 0;
        outfile << "\"";
   //     while( !inFileLine.eof() )
   //     {
   //         if( cnt > 0 )
   //             outfile << "_";
			//inFileLine >> temp;
   //         outfile << temp;
			//++cnt;
   //     }
        outfile << fileLine;
		outfile << "\"";

        outfile << "," << '\n';
    }
}