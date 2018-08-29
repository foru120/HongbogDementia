package com.hongbog.dementia.hongbogdementia;

import android.graphics.Bitmap;
import android.support.v4.os.TraceCompat;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.Vector;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_8UC1;

public class TensorFlowClassifier extends TensorFlowClassifierAbstract{
    private final String TAG = "TFClassifier";

    private final String[] INPUT_NAMES = {"model/input_module/x"};
    private final String[] OUTPUT_NAMES = {"model/output_module/softmax","grad_cam/outputs"};
    private final int WIDTH = 224;
    private final int HEIGHT = 224;
    private final String MODEL_FILE = "file:///android_asset/model_graph.pb";
    private final String LABEL_FILE = "file:///android_asset/label_strings.txt";

    private Vector<String> labels = new Vector<>();  // label 정보
    private float[] logits = new float[numClasses];  // logit 정보
    private int[] camSize = new int[]{7, 7};
    private float[] cam_outputs = new float[6 * camSize[0] * camSize[1]];

    private TensorFlowClassifier() {

    }

    private static class SingleToneHolder {
        static final TensorFlowClassifier instance = new TensorFlowClassifier();
    }

    public static TensorFlowClassifier getInstance() {
        return SingleToneHolder.instance;
    }

    @Override
    void dementiaDiagnosis(float[] imgData) {
        TraceCompat.beginSection("dementiaDiagnosis");

        TraceCompat.beginSection("feed");
        tii.feed(this.INPUT_NAMES[0], imgData, 1, this.HEIGHT, this.WIDTH, 3);
        TraceCompat.endSection();

        TraceCompat.beginSection("run");
        tii.run(this.OUTPUT_NAMES, false);
        TraceCompat.endSection();

        TraceCompat.beginSection("fetch");
        tii.fetch(this.OUTPUT_NAMES[0], this.logits);
        tii.fetch(this.OUTPUT_NAMES[1], this.cam_outputs);
        TraceCompat.endSection();
    }

    @Override
    float[] normalize(Bitmap bitmap) {
        int mWidth = bitmap.getWidth();
        int mHeight = bitmap.getHeight();

        int[] ori_pixels = new int[mWidth * mHeight];
        float[] norm_pixels = new float[mWidth * mHeight * 3];

        bitmap.getPixels(ori_pixels, 0, mWidth, 0, 0, mWidth, mHeight);
        for (int i = 0; i < ori_pixels.length; i++) {
            int R = (ori_pixels[i] >> 16) & 0xff;
            int G = (ori_pixels[i] >> 8) & 0xff;
            int B = ori_pixels[i] & 0xff;

            norm_pixels[(i * 3) + 0] = (float) R / 255.0f;
            norm_pixels[(i * 3) + 1] = (float) G / 255.0f;
            norm_pixels[(i * 3) + 2] = (float) B / 255.0f;
        }
        return norm_pixels;
    }

    public Bitmap gradcamVisualization(Bitmap oriBitmap) {
        oriBitmap = Bitmap.createScaledBitmap(oriBitmap, this.WIDTH, this.HEIGHT, false);

        Size cam_size = new Size(this.camSize[0], this.camSize[1]);
        Size img_size = new Size(this.WIDTH, this.HEIGHT);

        // Original 이미지
        Mat oriMat = new Mat(img_size, CV_32F);
        Utils.bitmapToMat(oriBitmap, oriMat);

        // CAM 출력 값
        Mat camMat = new Mat(cam_size, CV_32F);
        camMat.put(0, 0, this.cam_outputs);
        Imgproc.resize(camMat, camMat, img_size);

        camMat.convertTo(camMat, CV_8UC1);
        Imgproc.applyColorMap(camMat, camMat, Imgproc.COLORMAP_JET);
        Imgproc.cvtColor(camMat, camMat,  Imgproc.COLOR_BGR2RGBA);

        camMat.convertTo(camMat, CV_32F, 0.35);
        Imgproc.accumulate(oriMat, camMat);

        camMat.convertTo(camMat, CV_8UC1);

        Utils.matToBitmap(camMat, oriBitmap);
        return oriBitmap;
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
