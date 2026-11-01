#ifndef IntAndData_HPP
#define IntAndData_HPP


#include "cusort.hpp"
#include "valgen.hpp"


struct IntAndData{
    int val;
    int data;

    __host__ __device__ IntAndData& operator=(const int& v){
        val = v;
        data = 0; // Integer-generated records must have a defined payload.
        return *this;
    }

    // Operators for comparison, needed for sorting
    __device__ __host__ bool operator<(const IntAndData& other) const {
        return val < other.val;
    }
    __device__ __host__ bool operator<=(const IntAndData& other) const {
        return val <= other.val;
    }
    __device__ __host__ bool operator>(const IntAndData& other) const {
        return val > other.val;
    }
    __device__ __host__ bool operator>=(const IntAndData& other) const {
        return val >= other.val;
    }

    // Operator for stream <<
    __host__ friend std::ostream& operator<<(std::ostream& os, const IntAndData& obj) {
        os << "{" << obj.val << ", " << obj.data << "}";
        return os;
    }

    // Operator for equality, needed for verification
    __host__ __device__ bool operator==(const IntAndData& other) const {
        return val == other.val && data == other.data;
    }
    __host__ __device__ bool operator!=(const IntAndData& other) const {
        return !(*this == other);
    }

    static __host__ __device__ IntAndData max(){
        IntAndData res;
        res.val = my_numeric_limits_max<int>();
        res.data = my_numeric_limits_max<int>();
        return res;
    }

    static __host__ __device__ IntAndData min(){
        IntAndData res;
        res.val = my_numeric_limits_min<int>();
        res.data = my_numeric_limits_min<int>();
        return res;
    }

    __host__ __device__ IntAndData nextVal() const {
        IntAndData res = *this;
        res.val = getNextVal(res.val);
        return res;
    }

    __host__ __device__ IntAndData prevVal() const {
        IntAndData res = *this;
        res.val = getPrevVal(res.val);
        return res;
    }
};

    // operator for swapping, needed for sorting
__device__ __host__ void swap(IntAndData& a, IntAndData& b) {
    IntAndData temp = a;
    a = b;
    b = temp;
}


#endif