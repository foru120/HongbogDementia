package com.hongbog.dementia.hongbogdementia;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Vector;

public abstract class TensorFlowClassifierAbstract {
    TensorFlowInferenceInterface tii;
    int numClasses = 2;  // 최종 출력 class 개수

    abstract void dementiaDiagnosis(float[] imgData);

    abstract float[] normalize(final Bitmap bitmap);

    void createClassifier(final AssetManager assetManager, String LABEL_FILE, String MODEL_FILE,
                          Vector<String> labels, String TAG) {
        BufferedReader br = null;
        try {
            String actualFilename = LABEL_FILE.split("file:///android_asset/")[1];
            br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
            String line = "";
            while((line = br.readLine()) != null) {
                labels.add(line);
            }
        } catch (IOException e) {
            Log.d(TAG, e.toString());
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.d(TAG, e.toString());
            }
        }

        this.tii = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);
    }

    String getClassificationLabel(final float[] logits) {
        float maxValue = -1;
        int maxIndex = -1;

        for (int i = 0; i < this.numClasses; i++) {
            if (maxValue < logits[i]) {
                maxValue = logits[i];
                maxIndex = i;
            }
        }

        String result_txt;

        if (maxIndex == 0) {
            result_txt = "Normal";
        } else {
            result_txt = "Dementia";
        }

        return result_txt;
    }

    int getClassificationPercentage(final float[] logits){
        float maxValue = -1;

        for (int i = 0; i < this.numClasses; i++) {
            if (maxValue < logits[i]) {
                maxValue = logits[i];
            }
        }

        return Math.round(maxValue * 100);
    }
}
