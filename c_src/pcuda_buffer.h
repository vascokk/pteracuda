#ifndef PCUDA_BUFFER
#define PCUDA_BUFFER

#include <string>
#include <vector>

#include "erl_nif.h"

enum MatrixOrientation {
    ROW_MAJOR,
    COLUMN_MAJOR
};

enum PCudaBufferTypes {
    BUF_TYPE_INTEGER,
    BUF_TYPE_STRING,
    BUF_TYPE_FLOAT,
    BUF_TYPE_MATRIX_INTEGER,
    BUF_TYPE_MATRIX_FLOAT
};


class PCudaBuffer {
public:
    virtual ~PCudaBuffer() { };
    virtual unsigned int size() = 0;
    virtual PCudaBufferTypes type() = 0;
    virtual bool sort() = 0;
    virtual bool contains(ErlNifEnv *env, ERL_NIF_TERM rawTarget) = 0;
    virtual void write(ErlNifEnv *env, ERL_NIF_TERM data) = 0;
    virtual void delete_at(unsigned long position) = 0;
    virtual bool insert_at(unsigned long position, ErlNifEnv *env, ERL_NIF_TERM value) = 0;
    virtual void clear() = 0;
    virtual bool copy(PCudaBuffer *src) = 0;
    virtual ERL_NIF_TERM intersect(ErlNifEnv *env, PCudaBuffer *other) = 0;
    virtual ERL_NIF_TERM minmax(ErlNifEnv *env) = 0;
    virtual ERL_NIF_TERM toErlTerms(ErlNifEnv *env) = 0;    
};


class PCudaIntBuffer : public PCudaBuffer {
public:
    PCudaIntBuffer();
    PCudaIntBuffer(unsigned long size);
    virtual ~PCudaIntBuffer();
    virtual unsigned int size();
    virtual PCudaBufferTypes type() { return BUF_TYPE_INTEGER; };
    virtual bool sort();
    virtual bool contains(ErlNifEnv *env, ERL_NIF_TERM rawTarget);
    virtual ERL_NIF_TERM toErlTerms(ErlNifEnv *env);
    virtual void write(ErlNifEnv *env, ERL_NIF_TERM data);
    virtual void delete_at(unsigned long position);
    virtual bool insert_at(unsigned long position, ErlNifEnv *env, ERL_NIF_TERM value);
    virtual void clear();
    virtual bool copy(PCudaBuffer *src);
    virtual ERL_NIF_TERM intersect(ErlNifEnv *env, PCudaBuffer *other);
    virtual ERL_NIF_TERM minmax(ErlNifEnv *env);
    std::vector<long>* get_data(){return data;}; 
protected:
    std::vector<long> *data;
};

class PCudaFloatBuffer : public PCudaBuffer{
public:
    PCudaFloatBuffer();
    PCudaFloatBuffer(unsigned long size);
    virtual ~PCudaFloatBuffer();
    virtual unsigned int size();
    virtual PCudaBufferTypes type() { return BUF_TYPE_FLOAT; };
    virtual bool sort();
    virtual bool contains(ErlNifEnv *env, ERL_NIF_TERM rawTarget);
    virtual ERL_NIF_TERM toErlTerms(ErlNifEnv *env);
    virtual void write(ErlNifEnv *env, ERL_NIF_TERM data);
    virtual void delete_at(unsigned long position);
    virtual bool insert_at(unsigned long position, ErlNifEnv *env, ERL_NIF_TERM value);
    virtual void clear();
    virtual bool copy(PCudaBuffer *src);
    virtual ERL_NIF_TERM intersect(ErlNifEnv *env, PCudaBuffer *other);
    virtual ERL_NIF_TERM minmax(ErlNifEnv *env);
    std::vector<double>* get_data(){return data;};
protected:
    std::vector<double> *data;
};

class PCudaStringBuffer : public PCudaBuffer {
public:
    PCudaStringBuffer();
    virtual ~PCudaStringBuffer();
    virtual unsigned int size();
    virtual PCudaBufferTypes type() { return BUF_TYPE_STRING; };
    virtual bool sort();
    virtual bool contains(ErlNifEnv *env, ERL_NIF_TERM rawTarget);
    virtual ERL_NIF_TERM toErlTerms(ErlNifEnv *env);
    virtual void write(ErlNifEnv *env, ERL_NIF_TERM data);
    virtual void delete_at(unsigned long position);
    virtual bool insert_at(unsigned long position, ErlNifEnv *env, ERL_NIF_TERM value);
    virtual void clear();
    virtual bool copy(PCudaBuffer *src);
    virtual ERL_NIF_TERM intersect(ErlNifEnv *env, PCudaBuffer *other);
    virtual ERL_NIF_TERM minmax(ErlNifEnv *env) { return enif_make_atom(env, "error"); };
    std::vector<std::string>* get_data(){return data;};  
protected:
    std::vector<std::string> *data;
};

class PCudaMatrix {
public:
    unsigned int rows() {return _rows;};
    unsigned int cols() {return _cols;};
    MatrixOrientation storage() {return orientation;}
protected:
    unsigned int _rows;
    unsigned int _cols;
    MatrixOrientation orientation;
};


class PCudaMatrixIntBuffer : public PCudaMatrix, public PCudaIntBuffer {
public:
    PCudaMatrixIntBuffer();
    PCudaMatrixIntBuffer(unsigned int rows, unsigned int cols, MatrixOrientation orientation);
    virtual ~PCudaMatrixIntBuffer();
    virtual unsigned int size();
    virtual PCudaBufferTypes type() { return BUF_TYPE_MATRIX_INTEGER; };
    virtual ERL_NIF_TERM toErlTerms(ErlNifEnv *env);
    virtual void write(ErlNifEnv *env, ERL_NIF_TERM data);
    virtual void clear();
};

class PCudaMatrixFloatBuffer : public PCudaMatrix, public PCudaFloatBuffer {
public:
    PCudaMatrixFloatBuffer();
    PCudaMatrixFloatBuffer(unsigned int rows, unsigned int cols, MatrixOrientation orientation);
    virtual ~PCudaMatrixFloatBuffer();
    virtual unsigned int size();
    virtual PCudaBufferTypes type() { return BUF_TYPE_MATRIX_FLOAT; };
    virtual ERL_NIF_TERM toErlTerms(ErlNifEnv *env);
    virtual void write(ErlNifEnv *env, ERL_NIF_TERM data);
    virtual void clear();
};

#endif
