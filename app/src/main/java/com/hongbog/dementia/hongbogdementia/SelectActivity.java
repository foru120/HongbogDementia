package com.hongbog.dementia.hongbogdementia;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class SelectActivity extends AppCompatActivity {
    private static final String TAG = "SelectActivity";
    private static final int PICTURE_REQUEST_CODE = 100;

    public ImageView imgOriginal;
    public ImageView imgCAM;
    private TextView txtDisease;
//    private TextView txtAccuracy;

    private Button btnGallery;
    private Button btnDiagnosis;

    private Bitmap tmpBitmap;

    private Executor executor = Executors.newSingleThreadExecutor();

    private TensorFlowClassifier classifier = TensorFlowClassifier.getInstance();

    static {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_select);

        imgOriginal = (ImageView) findViewById(R.id.imgOriginal);
        imgCAM = (ImageView) findViewById(R.id.imgCAM);
        txtDisease = (TextView) findViewById(R.id.diseaseTxt);
//        txtAccuracy = (TextView) findViewById(R.id.accuracyTxt);

        btnGallery = (Button) findViewById(R.id.btnGallery);
        btnDiagnosis = (Button) findViewById(R.id.btnDiagnosis);

        btnGallery.setOnClickListener(new ButtonEventHandler());
        btnDiagnosis.setOnClickListener(new ButtonEventHandler());

        initTensorFlowAndLoadModel();
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try{
                    classifier.createClassifier(
                            getAssets(),
                            classifier.getLABEL_FILE(), classifier.getMODEL_FILE(),
                            classifier.getLabels(), classifier.getTAG());
                    Log.d(TAG, "Load Success");
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    public class ButtonEventHandler implements View.OnClickListener {
        @Override
        public void onClick(View v) {
            int objectID = v.getId();

            if (objectID == R.id.btnGallery) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Audio.Media.EXTERNAL_CONTENT_URI);
                intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, false);
                intent.setType("image/*");
                startActivityForResult(Intent.createChooser(intent, "Select Image"), PICTURE_REQUEST_CODE);
            } else if (objectID == R.id.btnDiagnosis) {
                Bitmap bitmapTmp = Bitmap.createScaledBitmap(tmpBitmap, classifier.getWIDTH(), classifier.getHEIGHT(), false);
                classifier.dementiaDiagnosis(classifier.normalize(bitmapTmp));
                String label = classifier.getClassificationLabel(classifier.getLogits());
//                int percentage = classifier.getClassificationPercentage(classifier.getLogits());
                Bitmap camBitmap = classifier.gradcamVisualization(tmpBitmap);
                imgCAM.setImageBitmap(camBitmap);
                txtDisease.setText(label);
//                txtAccuracy.setText(percentage + "%");
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICTURE_REQUEST_CODE) {
            if (resultCode == RESULT_OK) {
                imgOriginal.setImageResource(0);

                Uri uri = data.getData();

                if (uri != null) {
                    imgOriginal.setImageURI(uri);
                    Log.d("isroot--uri", String.valueOf(uri));
                    Bitmap bitmapOriginal = ((BitmapDrawable) imgOriginal.getDrawable()).getBitmap();
                    tmpBitmap = bitmapOriginal;
                }
            }
        }
    }
}