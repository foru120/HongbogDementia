package com.hongbog.dementia.hongbogdementia;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.support.v4.os.TraceCompat;

import java.util.Vector;

public class TensorFlowNewClassifier extends TensorFlowClassifierAbstract{
    private static final String TAG = "TensorFlowNewClassifier";

    private static final String[] INPUT_NAMES = {"model/input_module/x"};
    private static final String[] OUTPUT_NAMES = {"model/output_module/softmax"};
    private static final int WIDTH = 400;
    private static final int HEIGHT = 224;
    private static final String MODEL_FILE = "file:///android_asset/new_model_graph.pb";
    private static final String LABEL_FILE = "file:///android_asset/label_strings.txt";

    private Vector<String> labels = new Vector<>();  // label 정보
    private float[] logits = new float[numClasses];  // logit 정보

    private TensorFlowNewClassifier() {

    }

    private static class SingleToneHolder {
        static final TensorFlowNewClassifier instance = new TensorFlowNewClassifier();
    }

    public static TensorFlowNewClassifier getInstance() {
        return SingleToneHolder.instance;
    }

    @Override
    void dementiaDiagnosis(float[] imgData) {
        TraceCompat.beginSection("dementiaDiagnosis");

        TraceCompat.beginSection("feed");
        tii.feed(this.INPUT_NAMES[0], imgData, 1, this.HEIGHT, this.WIDTH, 1);
        TraceCompat.endSection();

        TraceCompat.beginSection("run");
        tii.run(this.OUTPUT_NAMES, false);
        TraceCompat.endSection();

        TraceCompat.beginSection("fetch");
        tii.fetch(this.OUTPUT_NAMES[0], this.logits);
        TraceCompat.endSection();
    }

    @Override
    public float[] normalize(final Bitmap bitmap) {
        int mWidth = bitmap.getWidth();
        int mHeight = bitmap.getHeight();

        int[] ori_pixels = new int[mWidth * mHeight];
        float[] norm_pixels = new float[mWidth * mHeight];

        bitmap.getPixels(ori_pixels, 0, mWidth, 0, 0, mWidth, mHeight);

        for (int i=0; i< ori_pixels.length; i++) {
            int grayPixel = (int) ((Color.red(ori_pixels[i]) * 0.299) + (Color.green(ori_pixels[i]) * 0.587) + (Color.blue(ori_pixels[i]) * 0.114));  // decode_png -> grayscale 변환과 일치
            if (grayPixel < 0) grayPixel = 0;
            if (grayPixel > 255) grayPixel = 255;
            norm_pixels[i] = grayPixel / 255.0f;
        }
        return norm_pixels;
    }

    public int getWIDTH() {
        return WIDTH;
    }

    public int getHEIGHT() {
        return HEIGHT;
    }

    public String getTAG() {
        return TAG;
    }

    public String getMODEL_FILE() {
        return MODEL_FILE;
    }

    public String getLABEL_FILE() {
        return LABEL_FILE;
    }

    public Vector<String> getLabels() {
        return labels;
    }

    public float[] getLogits() {
        return logits;
    }
}
