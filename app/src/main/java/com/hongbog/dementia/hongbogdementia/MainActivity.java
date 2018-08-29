package com.hongbog.dementia.hongbogdementia;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {
    private Button captureBtn;
    private Button selectBtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);
        captureBtn = (Button)findViewById(R.id.captureActivityBtn);
        captureBtn.setOnClickListener(new View.OnClickListener()
                                      {
                                          public void onClick(View v)
                                          {
                                              Intent intent = new Intent(MainActivity.this, CaptureActivity.class);
                                              startActivity(intent);
                                          }
                                      }
        );
        selectBtn = (Button)findViewById(R.id.selectActivityBtn);
        selectBtn.setOnClickListener(new View.OnClickListener()
                                      {
                                          public void onClick(View v)
                                          {
                                              Intent intent = new Intent(MainActivity.this, SelectActivity.class);
                                              startActivity(intent);
                                          }
                                      }
        );
    }
}
