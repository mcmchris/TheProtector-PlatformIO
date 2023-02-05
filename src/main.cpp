/*
The Protector firmware  (MUST BE COMPILED AND FLASHED ON PLATFORMIO)

@Author: Christopher Mendez
@Date: 02/02/2023 (mm/dd/yy)

@Brief:
This code runs and Edge Impulse model for fall detection, send temperature, humidity, pressure, IAQ, CO2, activity
and inference result data through BLE to be viewed and recorded on "The Protector" mobile app.

@Note:
The Edge Impulse model library of your project must be in the ./lib folder.

@Tutorial:
Arduino Projects Hub link: 

*/

/*
BLE comunication protocol:

1 -- temp, hum, press
2 -- Iaq, Co2, Activity(256-16384)
3 -- Inference(still,walking,falling)

Example of data to be sent (one line at a time):

1,25.32,45.2,994.2
2,24,500,256
3,walking

*/

/* Activities -- activ.value data:

256 = Still activity started.
512 = Walking activity started.
1024 = Running activity started.
2048 = Bicycle activity started.
4096 = Vehicle activity started.
8192 = Tilting activity started.
16384 = Vehicle still.

*/

/*
Espected inference results:

1 -- still
2 -- falling
3 -- walking

*/

#include "Nicla_System.h"              // Nicla library to be able to enable battery charging and LED control
#include "Arduino_BHY2.h"              // Arduino library to read data from every built-in sensor of the Nicla Sense ME
#include <ArduinoBLE.h>                // Bluetooth® Low Energy library for the Nicla Sense Me
#include <the_protector_inferencing.h> // Edge Impulse Arduino Library of your trained model

#define CONVERT_G_TO_MS2 9.80665f
#define FREQUENCY_HZ 10
#define INTERVAL_MS (1000 / (FREQUENCY_HZ + 1))
static unsigned long last_interval_ms = 0;
static unsigned long last_interval2_ms = 0;

/* Sensors class and IDs declaration */
SensorXYZ accel(SENSOR_ID_ACC);     // accelerometer
SensorBSEC bsec(SENSOR_ID_BSEC);    // 4-in-1 sensor for IAQ and CO2.
Sensor temp(SENSOR_ID_TEMP);        // temperature
Sensor baro(SENSOR_ID_BARO);        // pressure
SensorActivity activ(SENSOR_ID_AR); // activity detection

// To classify 1 frame of data you need EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE values
float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
// Keep track of where we are in the feature array
size_t feature_ix = 0;

// Bluetooth® Low Energy Service
BLEService myService("2d0a0000-e0f7-5fc8-b50f-05e267afeb67");

// Bluetooth® Low Energy Characteristic
BLEStringCharacteristic myChar("2d0a0001-e0f7-5fc8-b50f-05e267afeb67", // standard 16-bit characteristic UUID
                               BLERead | BLENotify, 56);               // remote clients will be able to get notifications if this characteristic changes

// RX characteristic and allow remote device to write data to the Nicla (for LED control)
BLEByteCharacteristic myCharRX("2d0a0002-e0f7-5fc8-b50f-05e267afeb67", BLERead | BLEWrite);

bool LED = false;
int oldIAQ = 0;

/* myCharRX characteristic handler (this will happen when something is received from the app to the board) */
void myCharRXWritten(BLEDevice central, BLECharacteristic characteristic)
{
  // Central wrote new value to characteristic, update LED
  if (myCharRX.value())
  {
    LED = true;
  }
  else
  {
    LED = false;
  }
}

void setup()
{
  // Initialize serial:
  Serial.begin(115200);

  /* Init Nicla system and enable battery charging (100mA) */
  nicla::begin();
  nicla::leds.begin();
  nicla::enableCharge(100);

  /* Init & start sensors */
  BHY2.begin(NICLA_I2C);
  accel.begin();
  baro.begin();
  bsec.begin();
  activ.begin();

  // BLE initialization
  if (!BLE.begin())
  {
    Serial.println("starting BLE failed!");

    while (1)
      ;
  }

  /* Set a local name for the Bluetooth® Low Energy device
     This name will appear in advertising packets
     and can be used by remote devices to identify this Bluetooth® Low Energy device
     The name can be changed but maybe be truncated based on space left in advertisement packet
  */
  BLE.setLocalName("The_Protector");
  BLE.setAdvertisedService(myService);   // add the service UUID
  myService.addCharacteristic(myChar);   // add the sensor data characteristic
  myService.addCharacteristic(myCharRX); // add the BLE receive data characteristic

  BLE.addService(myService); // Add the custom service

  // Assign event handlers for characteristic
  myCharRX.setEventHandler(BLEWritten, myCharRXWritten);

  // Start advertising
  BLE.advertise();

  delay(1500);
}

void loop()
{

  BLEDevice central = BLE.central();
  String inferenceResult = "";

  // If a central is connected to the peripheral:
  if (central)
  {

    nicla::leds.setColor(blue);
    delay(100);
    nicla::leds.setColor(off);
    delay(100);
    nicla::leds.setColor(blue);
    delay(100);
    nicla::leds.setColor(off);

    // Run machine learning model
    // while the central is connected:
    while (central.connected())
    {

      if (millis() > last_interval_ms + INTERVAL_MS)
      {
        last_interval_ms = millis();

        BHY2.update(); // update the sensors

        // Get accel data
        short accX, accY, accZ;
        accX = (accel.x() * 8.0 / 32768.0) * CONVERT_G_TO_MS2;
        accY = (accel.y() * 8.0 / 32768.0) * CONVERT_G_TO_MS2;
        accZ = (accel.z() * 8.0 / 32768.0) * CONVERT_G_TO_MS2;

        // Fill the features buffer
        features[feature_ix++] = accX;
        features[feature_ix++] = accY;
        features[feature_ix++] = accZ;

        // Features buffer full? then classify!
        if (feature_ix == EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE)
        {
          ei_impulse_result_t result;

          // Create signal from features frame
          signal_t signal;
          numpy::signal_from_buffer(features, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);

          // Run classifier
          EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);
          ei_printf("run_classifier returned: %d\n", res);
          if (res != 0)
          {
            feature_ix = 0;
            return;
          }

          // Print predictions
          ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
                    result.timing.dsp, result.timing.classification, result.timing.anomaly);
          // Store the location of the highest classified label

          for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++)
          {

            ei_printf("%s:\t%.5f\n", result.classification[ix].label, result.classification[ix].value);

            if (result.classification[ix].value > 0.7 && result.anomaly > 0.7 && inferenceResult != result.classification[ix].label)
            {
              inferenceResult = result.classification[ix].label;
            }
          }

          // Reset features frame
          feature_ix = 0;
        }
      }

      /* Send sensor data every 1000 ms */
      long currentMillis = millis();

      if (currentMillis - last_interval2_ms >= 1000)
      {
        last_interval2_ms = currentMillis;

        String msg = "1," + String(bsec.comp_t() - 6, 1) + "," + String(bsec.comp_h(), 1) + "," + String(baro.value(), 1);
        String msg2 = "2,";
        oldIAQ = bsec.iaq();
        msg2 += oldIAQ;
        msg2 += ",";
        msg2 += bsec.co2_eq();
        msg2 += ",";
        msg2 += activ.value();
        String msg3 = "3,";
        msg3 += inferenceResult;

        myChar.writeValue(msg); // send line 1
        delay(100);
        myChar.writeValue(msg2); // send line 2
        delay(100);
        myChar.writeValue(msg3); // send line 3
      }

      /* Control the LED and show IAQ status with colors */
      if (LED)
      {
        if (oldIAQ >= 0 && oldIAQ <= 100)
        {
          nicla::leds.setColor(0, 255, 0);
        }
        if (oldIAQ >= 101 && oldIAQ <= 250)
        {
          nicla::leds.setColor(255, 255, 0);
        }
        if (oldIAQ >= 251)
        {
          nicla::leds.setColor(255, 0, 0);
        }
      }
      else
      {
        nicla::leds.setColor(off);
      }
    }
  }

  nicla::leds.setColor(off); // turn off the LED if Bluetooth is disconnected
}
