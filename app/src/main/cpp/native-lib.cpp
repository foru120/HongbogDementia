#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;


extern "C"
JNIEXPORT void JNICALL
Java_com_hongbog_dementia_hongbogdementia_MainActivity_Divied(JNIEnv *env, jobject instance,
                                                              jlong matAddrInput,
                                                              jlong matAddrResult) {

    // TODO
    Mat &matInput = *(Mat *)matAddrInput;
    Mat &matResult = *(Mat *)matAddrResult;

    //divide(matInput, matResult, matResult);
    Mat mat255(matInput.size(), CV_32F, Scalar(255.0));

    divide(matInput, mat255, matResult);

}

extern "C"
JNIEXPORT void JNICALL
Java_com_hongbog_dementia_hongbogdementia_MainActivity_Nomal(JNIEnv *env, jobject instance,
                                                             jlong matAddrInput,
                                                             jlong matAddrResult) {

    // TODO
    Mat &matInput = *(Mat *)matAddrInput;
    Mat &matResult = *(Mat *)matAddrResult;

    double minVal, maxVal;
    minMaxIdx(matInput, &minVal, &maxVal);

    Mat matMax(matInput.size(), CV_32F, Scalar(maxVal));    //output.max()

    divide(matInput, matMax, matResult);                    //  output /= output.max()

}