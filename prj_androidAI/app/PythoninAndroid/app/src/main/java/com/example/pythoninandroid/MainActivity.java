package com.example.pythoninandroid;

import android.os.Bundle;
import android.view.TextureView;
import android.view.View;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.chaquo.python.PyException;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

public class MainActivity extends AppCompatActivity {
    private TextView textViewOutput;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(getApplicationContext()));
        }

        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.textViewOutput), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);

            textViewOutput = findViewById(R.id.textView);
            return insets;
        });
    }

    public void buttonPythonRun(View view){
        Python python = Python.getInstance();

        try (PyObject pyObject = python.getModule("add_nums")
                .callAttr("add_nums", 10, 20)) {
            textViewOutput.setText(pyObject.toString());
        } catch (PyException e) {
            textViewOutput.setText(e.toString());
        }

    }
}