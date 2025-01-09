package com.example.pythoninandroid;

import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import com.chaquo.python.PyException;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

public class MainActivity extends AppCompatActivity {
    private TextView textViewOutput;
    private TextView textViewResult; // New TextView for the result
    private EditText editTextNumber1;
    private EditText editTextNumber2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(getApplicationContext()));
        }

        setContentView(R.layout.activity_main);

        textViewOutput = findViewById(R.id.textView); // This is for the title
        textViewResult = findViewById(R.id.textViewResult); // This is for the result
        editTextNumber1 = findViewById(R.id.editTextNumber1);
        editTextNumber2 = findViewById(R.id.editTextNumber2);
    }

    public void buttonPythonRun(View view){
        Python python = Python.getInstance();
        String result; // Store the result

        try {
            int num1 = Integer.parseInt(editTextNumber1.getText().toString());
            int num2 = Integer.parseInt(editTextNumber2.getText().toString());

            try (PyObject pyObject = python.getModule("add_nums")
                    .callAttr("add_nums", num1, num2)) {
                result = pyObject.toString(); // Store the result
            } catch (PyException e) {
                result = e.toString(); // Store the error message
            }

        } catch (NumberFormatException e) {
            result = "Please enter valid numbers"; // Store the error message
        }

        textViewResult.setText(result); // Display the result in the new TextView
    }
}