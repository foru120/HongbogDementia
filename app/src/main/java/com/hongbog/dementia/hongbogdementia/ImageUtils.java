/*
 * Copyright 2016-present Tzutalin
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.hongbog.dementia.hongbogdementia;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.support.annotation.Keep;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;


/**
 * Utility class for manipulating images.
 **/
public class ImageUtils {
    private static final String TAG = ImageUtils.class.getSimpleName();

    /**
     * Saves a Bitmap object to disk for analysis.
     * @param bitmap The bitmap to save.
     */
    public static void saveBitmap(final Bitmap bitmap, String stringTime) {
        final String root =
                Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "dlib";
//        Timber.tag(TAG).d(String.format("Saving %dx%d bitmap to %s.", bitmap.getWidth(), bitmap.getHeight(), root));
        final File myDir = new File(root);

        if (!myDir.mkdirs()) {
//            Timber.tag(TAG).e("Make dir failed");
        }
        // file name
        String fname = ".png";
        Date today = new Date();
        String str = today.toString();
        fname = stringTime + fname;
        //fname = str+"t"+ stringTime +fname;

        final File file = new File(myDir, fname);
        if (file.exists()) {
            file.delete();
        }
        try {
            final FileOutputStream out = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 99, out);
            out.flush();
            out.close();
        } catch (final Exception e) {
//            Timber.tag(TAG).e("Exception!", e);
        }
    }


    /**
     * Saves a Bitmap object to disk for analysis.
     * @param bitmap The bitmap to save.
     * @param stringTime file Name
     * @param dirName directory Name
     */
    public static void saveBitmap(final Bitmap bitmap, final String stringTime, final String dirName) {
        final String root =
                Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "dlib" + File.separator + dirName;
//        Timber.tag(TAG).d(String.format("Saving %dx%d bitmap to %s.", bitmap.getWidth(), bitmap.getHeight(), root));
        final File myDir = new File(root);

        if (!myDir.mkdirs()) {
//            Timber.tag(TAG).e("Make dir failed");
        }
        // file name
        String fname = ".png";
        Date today = new Date();
        String str = today.toString();
        fname = stringTime + fname;
        //fname = str+"t"+ stringTime +fname;

        final File file = new File(myDir, fname);
        if (file.exists()) {
            file.delete();
        }
        try {
            final FileOutputStream out = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 99, out);
            out.flush();
            out.close();
        } catch (final Exception e) {
//            Timber.tag(TAG).e("Exception!", e);
        }
    }


    /**
     * Saves a Bitmap object to disk for analysis.
     * @param dirName directory Name
     * @return extract Bitmap[] in dirName
     */
    public static ArrayList<Bitmap> extractBitmapsFromDirName(final String dirName) {
        final String root =
                Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "dlib" + File.separator + dirName;

        if (root == null) return null;

        final File myDir = new File(root);

        if(!myDir.exists()) return  null;

        ArrayList<Bitmap> bitmapArrayList = new ArrayList<>();

        for(File file : myDir.listFiles()){
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            Bitmap bitmap = BitmapFactory.decodeFile(file.getAbsolutePath(), options);
            bitmapArrayList.add(bitmap);
        }

        return bitmapArrayList;
    }


    /**
     * Saves a Bitmap object to disk for analysis.
     * @param filePath full file path
     * @return extract Bitmap from path
     */
    public static Bitmap extractBitmapFromDirName(String filePath){
        try {
            FileInputStream is = new FileInputStream(filePath);
            Bitmap bmp = BitmapFactory.decodeStream(is);
            is.close();
            return bmp;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
