#include <stdio.h>
#include <iostream>
#include "pcuda_buffer.h"
#include "pcuda_ops.h"

//from row,col to index, for a given leading dimension
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

//from row major to column major indexes
#define IDX2RRM(idx,C) (idx/C) //Index to row for RM matrix
#define IDX2CRM(idx,C) (idx%C) //index to col for RM matrix

// from column major to row major indexes
#define IDX2RCM(idx,R) (idx%R) //Idex to row for CM matrix
#define IDX2CCM(idx,R) (idx/R) //index to col for CM matrix

PCudaMatrixFloatBuffer::PCudaMatrixFloatBuffer():PCudaFloatBuffer() {
    this->_rows = 1;
    this->_cols = data->size();
    
}

PCudaMatrixFloatBuffer::PCudaMatrixFloatBuffer(unsigned int rows, unsigned int cols, MatrixOrientation orientation):PCudaFloatBuffer(rows*cols){
    this->_rows = rows;
    this->_cols = cols;
    this->orientation = orientation;
}

PCudaMatrixFloatBuffer::~PCudaMatrixFloatBuffer() {
}

unsigned int PCudaMatrixFloatBuffer::size() {
    return this->data->size();
}


void PCudaMatrixFloatBuffer::write(ErlNifEnv *env, ERL_NIF_TERM data) {
    ERL_NIF_TERM head_row;
    ERL_NIF_TERM head,tail;
    double value;
    long lvalue;

    unsigned ld = this->_rows; //number of rows; this is the lead dimension (ld) in row major matrices
    unsigned C = this->_cols; 
    unsigned long idx = 0;
   
    //CUBLAS uses column major matrices. this converts Erlang row major matrix ("list-of-lists") to column major one dimensional vector   
    if(this->orientation == ROW_MAJOR){
        while (enif_get_list_cell(env, data, &head_row, &data)) {
          while (enif_get_list_cell(env, head_row, &head, &head_row))
            if (enif_get_double(env, head, &value)) {
                this->data->at((IDX2C(IDX2RRM(idx,C), IDX2CRM(idx,C), ld)) ) = value;
                ++idx;
            }else if (enif_get_long(env, head, &lvalue)) {
                this->data->at(IDX2C(IDX2RRM(idx,C), IDX2CRM(idx,C), ld)) = (double)lvalue;
                ++idx;
            }
        }
        
        if(idx != this->data->size()){ 
            throw std::runtime_error("ERROR: Data does not fit the matrix size.");
        }
    } else {
        while (enif_get_list_cell(env, data, &head, &data)) {
            if (enif_get_double(env, head, &value)) {
                this->data->push_back(value);
            }else if (enif_get_long(env, head, &lvalue)) {
                this->data->push_back((double)lvalue);
            }
        }
    }

}


ERL_NIF_TERM PCudaMatrixFloatBuffer::toErlTerms(ErlNifEnv *env) {
    std::vector<double>::iterator iter;
    ERL_NIF_TERM retval = enif_make_list(env, 0);
    ERL_NIF_TERM row;

    unsigned R = this->_rows;
    unsigned Ridx;
    unsigned long idx = 0;
    std::vector<ERL_NIF_TERM> rows;

    if(this->rows() > 1 && this->orientation == ROW_MAJOR){
        for(int i=0; i<this->_rows; i++){
            row = enif_make_list(env, 0);
            rows.push_back(row);
        }
        
        for (iter = this->data->end(); iter != this->data->begin(); ) {
                --iter;
                Ridx = IDX2RCM(idx,R);
                rows[Ridx] = enif_make_list_cell(env, enif_make_double(env, (double)*iter), rows[Ridx]);
                ++idx;
        };
        for(int i=0; i<this->_rows; i++){
            retval = enif_make_list_cell(env, rows[i], retval);
        };
    }else{
        if (this->data->size() > 0) {
            for (iter = this->data->end(); iter != this->data->begin();) {
                --iter;
                retval = enif_make_list_cell(env, enif_make_double(env, (double)*iter), retval);
            }
        }        
    }

    return retval;
}

void PCudaMatrixFloatBuffer::clear() {
    this->data->clear();
}


