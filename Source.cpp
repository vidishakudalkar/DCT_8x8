#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;



Mat DCT(Mat image_hsi) {
	Mat image_dct(image_hsi.rows, image_hsi.cols, image_hsi.type());
	int height = image_hsi.rows - (image_hsi.rows % 8);
	int width = image_hsi.cols - (image_hsi.cols % 8);
	float temp;
	double alphai, alphaj;
	for (int currentrow = 0; currentrow < height; currentrow += 8)
	{
		for (int currentcolumn = 0; currentcolumn < width; currentcolumn += 8)
		{
			for (int i = currentrow; i < currentrow + 8; i++)
			{
				for (int j = currentcolumn; j < currentcolumn + 8; j++)
				{

					temp = 0.0;
					if (i == 0) alphai = 1.0 / sqrt(2.0); else alphai = 1.0;
					if (j == 0) alphaj = 1.0 / sqrt(2.0); else alphaj = 1.0;
					for (int x = currentrow; x < currentrow + 8; x++)
					{
						for (int y = currentcolumn; y < currentcolumn + 8; y++)
						{
							temp += (image_hsi.at<Vec3b>(x, y)[2])
								*(cos((((2 * x) + 1))
									* ((i *3.14159265)) / (2 * 8)))
								*(cos((((2 * y) + 1))
									* ((j *3.14159265)) / (2 * 8)));
						}
					}
					temp *= ((2 / sqrt(8 * 8)))*(alphai * alphaj);
					image_dct.at<Vec3b>(i, j)[2] = int(temp);
				}
			}
		}
	}

	return image_dct;

}

Mat DCT_DC(Mat dct) {

	int height = dct.rows - (dct.rows % 8);
	int width = dct.cols - (dct.cols % 8);
	Mat image_dct_dc(dct.rows, dct.cols, dct.type());
	image_dct_dc = dct.clone();
	for (int currentrow = 0; currentrow < height; currentrow += 8)
	{
		for (int currentcolumn = 0; currentcolumn < width; currentcolumn += 8)
		{
			for (int i = currentrow; i < currentrow + 8; i++)
			{
				for (int j = currentcolumn; j < currentcolumn + 8; j++)
				{

					for (int x = currentrow; x < currentrow + 8; x++)
					{
						for (int y = currentcolumn; y < currentcolumn + 8; y++)
						{
							if (i == currentrow && j == currentcolumn)
							{
								image_dct_dc.at<Vec3b>(i, j)[2] = image_dct_dc.at<Vec3b>(i, j)[2];
							}
							else
							{
								image_dct_dc.at<Vec3b>(i, j)[2] = 0;
							}
						}
					}

				}
			}
		}
	}
	return image_dct_dc;
}
Mat DCT_9_DC(Mat dct) {
	Mat freqimg(dct.rows, dct.cols, dct.type());
	Mat dcts(dct.rows, dct.cols, dct.type());
	Mat d1(dct.rows, dct.cols, dct.type());

	int height = dct.rows - (dct.rows % 8);
	int width = dct.cols - (dct.cols % 8);

	Mat image_dct_dc_9components(dct.rows, dct.cols, dct.type());
	image_dct_dc_9components = dct.clone();
	for (int currentrow = 0; currentrow < height; currentrow += 8)
	{
		for (int currentcolumn = 0; currentcolumn < width; currentcolumn += 8)
		{
			for (int i = currentrow; i < currentrow + 8; i++)
			{
				for (int j = currentcolumn; j < currentcolumn + 8; j++)
				{

					for (int x = currentrow; x < currentrow + 8; x++)
					{
						for (int y = currentcolumn; y < currentcolumn + 8; y++)
						{
							if (i == currentrow && j == currentcolumn || i == currentrow && j == currentcolumn + 1 || i == currentrow && j == currentcolumn + 2 || i == currentrow && j == currentcolumn + 3 || i == currentrow + 1 && j == currentcolumn || i == currentrow + 1 && j == currentcolumn + 1 || i == currentrow + 1 && j == currentcolumn + 2 || i == currentrow + 2 && j == currentcolumn || i == currentrow + 2 && j == currentcolumn + 1)
							{
								image_dct_dc_9components.at<Vec3b>(i, j)[2] = image_dct_dc_9components.at<Vec3b>(i, j)[2];
							}
							else
							{
								image_dct_dc_9components.at<Vec3b>(i, j)[2] = 0;
							}
						}
					}

				}
			}
		}
	}
	return image_dct_dc_9components;
}

Mat IDCT(Mat dct) {
	int height = dct.rows - (dct.rows % 8);
	int width = dct.cols - (dct.cols % 8);
	float temp;
	double alphai, alphaj;


	Mat image_idct_dc(dct.rows, dct.cols, dct.type());
	for (int currentrow = 0; currentrow < height; currentrow += 8)
	{
		for (int currentcolumn = 0; currentcolumn < width; currentcolumn += 8)
		{
			for (int i = currentrow; i < currentrow + 8; i++)
			{
				for (int j = currentcolumn; j < currentcolumn + 8; j++)
				{

					temp = 0.0;
					if (i == 0) alphai = 1.0 / sqrt(2.0); else alphai = 1.0;
					if (j == 0) alphaj = 1.0 / sqrt(2.0); else alphaj = 1.0;
					for (int x = currentrow; x < currentrow + 8; x++)
					{
						for (int y = currentcolumn; y < currentcolumn + 8; y++)
						{

							temp += (dct.at<Vec3b>(x, y)[2])
								*(cos((((2 * i) + 1)) * ((x * 3.14159)) / (2 * 8)))
								*(cos((((2 * j) + 1)) * ((y * 3.14159)) / (2 * 8)))
								*(alphai*alphaj);
						}
					}
					temp *= ((2 / sqrt(8 * 8)));
					image_idct_dc.at<Vec3b>(i, j)[2] = int(temp);
				}
			}
		}
	}


	return image_idct_dc;
}

Mat HSI(Mat image) {
	float red, green, blue, hue, saturation, intensity;
	Mat hsi_image(image.rows, image.cols, image.type());
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			blue = image.at<Vec3b>(i, j)[0];
			green = image.at<Vec3b>(i, j)[1];
			red = image.at<Vec3b>(i, j)[2];

			intensity = (blue + green + red) / 3;

			int min_val = 0;
			min_val = std::min(red, std::min(blue, green));

			saturation = 1 - 3 * (min_val / (blue + green + red));
			if (saturation < 0.00001)
			{
				saturation = 0;
			}
			else if (saturation > 0.99999) {
				saturation = 1;
			}

			if (saturation != 0)
			{
				hue = 0.5 * ((red - green) + (red - blue)) / sqrt(((red - green)*(red - green)) + ((red - blue)*(green - blue)));
				hue = acos(hue);

				if (blue <= green)
				{
					hue = hue;
				}
				else {
					hue = ((360 * 3.14159265) / 180.0) - hue;
				}
			}


			hsi_image.at<Vec3b>(i, j)[2] = intensity;

		}
	}
	return hsi_image;
}
void showoutput(string windowname, Mat image) {
	namedWindow(windowname, CV_WINDOW_AUTOSIZE);
	imshow(windowname, image);
	waitKey();
	cvDestroyAllWindows();
}
int main()
{
	Mat image;
	image = imread("basel3.bmp", CV_LOAD_IMAGE_COLOR);
	if (image.empty())
		cerr << "Error: Loading image" << endl;

	Mat image_hsi(image.rows, image.cols, image.type());
	Mat image_dct(image.rows, image.cols, image.type());
	Mat image_dct_dc(image.rows, image.cols, image.type());
	Mat image_dct_dc_9components(image.rows, image.cols, image.type());
	Mat image_idct_dc(image.rows, image.cols, image.type());
	Mat image_idct_dc_9components(image.rows, image.cols, image.type());

	image_hsi = HSI(image);
	namedWindow("HSI image", CV_WINDOW_AUTOSIZE);
	namedWindow("RGB image", CV_WINDOW_AUTOSIZE);
	imshow("HSI image", image_hsi);
	imshow("RGB image", image);
	waitKey();
	cvDestroyAllWindows();
	image_dct = DCT(image_hsi);
	showoutput("DCT IMAGE", image_dct);
	image_dct_dc = DCT_DC(image_dct);
	showoutput("DCT Only DC Components", image_dct_dc);
	image_dct_dc_9components = DCT_9_DC(image_dct);
	showoutput("DCT Only first 9 Low Frequency Components", image_dct_dc_9components);
	image_idct_dc = IDCT(image_dct_dc);
	showoutput("IDCT OF DC IMAGE", image_idct_dc);
	image_idct_dc_9components = IDCT(image_dct_dc_9components);
	showoutput("IDCT OF First 9 Low Frequency Components ", image_idct_dc_9components);


	waitKey(0);
	return 0;
}
