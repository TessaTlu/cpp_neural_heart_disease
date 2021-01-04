/*
В данном коде представлена реализация нейронной сети (многослойного перцептрона) обучение которой построено на базе алгоритма обратного распространения ошибки.
Сама нейросеть описана с помощью класса Neural. Применение её представлено на примере данных о результатах диагностики порока сердца.
Данные были взяты с форума по машинному обучениию Kaggle. Ссылка - https://www.kaggle.com/johnsmith88/heart-disease-dataset.
Нейросеть способна предсказать порок сердца с точностью в 84% (видно из консольного вывода). Это далеко не предел для этого набора данных,
однако такая точность свидетельствует о том, что нейросеть действительно обучена и способна использоваться по назначению.
*/
#include <iostream>
#include <vector>
#include <iomanip>      
#include <fstream>
#include <cstdlib>
#include <string>
using namespace std;
class Neural {		// Начинаем описывать класс 
public:
	int outlayer = 1;	// Размерность выходного слоя. Устанавливаем значение 1, так как в данном примере решается задача бинарной классификаци.
	int secondlayer = 13;	// Размеры скрытых слоёв можно устанавливать разные. От них отчасти зависит точность модели, но они не играют решающий роли.
	int firstlayer = 13;	// firstlayer и secondlayer - размеры скрытых скроёв
	int zerolayer = 13;		// zerolayer - размер входного входа. Это значение не случайное, оно предопределено количеством входных параметров
	double learnrate = 2.05;	// В ходе обучения, как было озвучено выше, используется алгоритм обратного распространения ошибки
								// Параметр learnrate определяет то, на каком локальном минимуме застопорится изменение веса связи.
								// данный параметр так же можно менять, от него напрямую зависит точность представленной нейросети. 
	vector<double> input_layer;			//	Каждый слой будет представлен в виде одномерного вектора
	vector<vector<double>> weights_0_1;	//	А веса связей между слоями - в виде двумерного вектора. 
	vector<double> first_layer;			//	Для упрощения представления веса связей - таблица, где двум нейронам соответствует один вес связи между ними
	vector<double> miss_first;			//	Объявляем все необходимые слои и веса связей
	vector<vector<double>> weights_1_2;	//	Также, обращаю внимание на то, что каждому слою соответствует вектор, который хранит в себе ошибки каждого нейрона слоя.
	vector<double> second_layer;		//	Эта мера позволяет реализовать алгоритм обратного распространения ошибки
	vector<double> miss_second;			//	Все векторы с началом "miss" в названии - то, о чём идёт речь.
	vector<vector<double>> miss_2_out;	//				
	vector<vector<double>> weights_2_out;// 
	vector<double> out_layer;
	vector<double> miss_out;
	Neural() {		// Опишем конструктор класса Neural
		input_layer.resize(zerolayer);	//	Устанавливаем всем слоям, весам связей и векторам, хранящим ошибки нейронов соотвтетсвующие размеры
		first_layer.resize(firstlayer);
		miss_first.resize(firstlayer);
		second_layer.resize(secondlayer);
		miss_second.resize(secondlayer);
		out_layer.resize(outlayer);
		miss_out.resize(outlayer);

		weights_0_1.resize(firstlayer, vector <double>(zerolayer));
		weights_1_2.resize(secondlayer, vector <double>(firstlayer));
		weights_2_out.resize(outlayer, vector <double>(secondlayer));
		// Перед тем как начать обучение нейросети необходимо задать весам связей стартовые значения
		// Они могут быть случайными, желательно в пределах от -1 до 1
		for (int i = 0; i < weights_0_1.size(); i++) {
			for (int j = 0; j < weights_0_1[0].size(); j++) {
				weights_0_1[i][j] = (1 + rand() % 200) / 100. - 1;
			}
		}
		for (int i = 0; i < weights_1_2.size(); i++) {
			for (int j = 0; j < weights_1_2[0].size(); j++) {
				weights_1_2[i][j] = (1 + rand() % 200) / 100. - 1;
			}
		}
		for (int i = 0; i < weights_2_out.size(); i++) {
			for (int j = 0; j < weights_2_out[0].size(); j++) {
				weights_2_out[i][j] = (1 + rand() % 200) / 100. - 1;
			}
		}
	}		// На этом инициализация нейросети окончена, переходим к описанию вспомогательных функций
	vector<double> matrix_vector(vector<vector<double>> matrix, vector<double> Vector) {
		vector <double> answer(matrix.size());	// Перемножение матрицы на вектор - базовая функция при работе с перцептронами
		double sum = 0;							// Алгоритм очевидный
		for (int i = 0; i < matrix.size(); i++) {
			for (int j = 0; j < matrix[0].size(); j++) {
				sum = sum + matrix[i][j] * Vector[j];
			}
			answer[i] = sum;
			sum = 0;
		}
		return answer;
	}
	vector <double> activation(vector<double> Vector) {	// Определим функцию активации нейрона, она необходима для того, 
		for (int i = 0; i < Vector.size(); i++) {		// чтобы ограничить возможные значения нейронов в пределах от 0 до 1 
			Vector[i] = 1 / (1 + exp(-Vector[i]));		// Для примера используется функция сигмоиды
		}
		return Vector;
	}
	vector<double> predict(vector<double> input) {		// Сердце нейросети - алгоритм осуществления прогноза
		input_layer = input;							// В сущности, вся работа нейросети - это последовательное перемножение весов связей на значение каждого нейрона
		input_layer = activation(input_layer);			// Алгоритм начинается с активации входного слоя
		first_layer = matrix_vector(weights_0_1, input_layer);	// Значения первого слоя определяются перемножением весов связей на значения входного слоя
		first_layer = activation(first_layer);					// После перемножения необходимо выполнить активацию 
		second_layer = matrix_vector(weights_1_2, first_layer);	// Далее этот алгоритм повторяется вплоть до выходного слоя
		second_layer = activation(second_layer);
		out_layer = matrix_vector(weights_2_out, second_layer);
		out_layer = activation(out_layer);
		return out_layer;
	}
	void learn_back_prop(double miss) {				// Функция, делающая нашу нейросеть живой			
		miss_out[0] = miss * out_layer[0] * (1 - out_layer[0]);	// В следующий циклах реализован алгоритм обратного распространения ошибки
		for (int j = 0; j < secondlayer; j++) {					// увидеть формулу из данных записей трудно, поэтому кратко опишу происходящее
			miss_second[j] = miss_out[0] * second_layer[j] * (1 - second_layer[j]) * weights_2_out[0][j];
		}														// 1) Сначала вычисляются ошибки каждого слоя
		for (int i = 0; i < firstlayer; i++) {
			miss_first[i] = 0;
			for (int j = 0; j < secondlayer; j++) {
				miss_first[i] = miss_first[i] + miss_second[j] * weights_1_2[j][i] * first_layer[i] * (1 - first_layer[i]);
			}
		}
		for (int i = 0; i < outlayer; i++) {
			for (int j = 0; j < secondlayer; j++) {
				weights_2_out[i][j] = weights_2_out[i][j] + miss_out[i] * second_layer[j] * learnrate;
			}
		}														// 2) Затем, на базе вычисленных ошибок изменяются значения весов связей
		for (int i = 0; i < secondlayer; i++) {
			for (int j = 0; j < firstlayer; j++) {
				weights_1_2[i][j] = weights_1_2[i][j] + miss_second[i] * first_layer[j] * learnrate;
			}
		}
		for (int i = 0; i < firstlayer; i++) {
			for (int j = 0; j < zerolayer; j++) {
				weights_0_1[i][j] = weights_0_1[i][j] + miss_first[i] * input_layer[j] * learnrate;
			}
		}
	}
};
vector <vector <double>> LoadHeartData() {	// Это функция, которая представляет csv файл в виде двумерного вектора
	ifstream ip("heart.csv");

	if (!ip.is_open()) std::cout << "ERROR: File Open" << '\n';

	string age;
	string sex;
	string cp;
	string trestbps;
	string chol;
	string fbs;
	string restecg;
	string thalach;
	string exang;
	string oldpeak;
	string slope;
	string ca;
	string thal;
	string target;
	vector <vector <double>> data;
	int rows_size = 0;
	while (ip.good()) {
		rows_size++;
		data.resize(rows_size, vector <double>(14));
		getline(ip, age, ',');
		data[rows_size - 1][0] = atof(age.c_str());
		getline(ip, sex, ',');
		data[rows_size - 1][1] = atof(sex.c_str());
		getline(ip, cp, ',');
		data[rows_size - 1][2] = atof(cp.c_str());
		getline(ip, trestbps, ',');
		data[rows_size - 1][3] = atof(trestbps.c_str());
		getline(ip, chol, ',');
		data[rows_size - 1][4] = atof(chol.c_str());
		getline(ip, fbs, ',');
		data[rows_size - 1][5] = atof(fbs.c_str());
		getline(ip, restecg, ',');
		data[rows_size - 1][6] = atof(restecg.c_str());
		getline(ip, thalach, ',');
		data[rows_size - 1][7] = atof(thalach.c_str());
		getline(ip, exang, ',');
		data[rows_size - 1][8] = atof(exang.c_str());
		getline(ip, oldpeak, ',');
		data[rows_size - 1][9] = atof(oldpeak.c_str());
		getline(ip, slope, ',');
		data[rows_size - 1][10] = atof(slope.c_str());
		getline(ip, ca, ',');
		data[rows_size - 1][11] = atof(ca.c_str());
		getline(ip, thal, ',');
		data[rows_size - 1][12] = atof(thal.c_str());
		getline(ip, target, '\n');
		data[rows_size - 1][13] = atof(target.c_str());

	}

	ip.close();
	return data;
}
int main() {
	Neural neural = Neural();	// Объявим объект класса и укажем конструктор для него

	vector <vector <double>> data = LoadHeartData();	// Загрузим данны из csv файла в вектор
	// Отделим анализы от диагноза:
	vector <vector <double>> X;
	X.resize(data.size(), vector<double>(data[0].size() - 1));
	vector <vector <double>> y;
	y.resize(data.size(), vector<double>(1));

	// В данном цикле мы разделим исходые данные на признаки X и ожидания y
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < data[0].size() - 1; j++) {
			X[i][j] = data[i][j];
		}
		y[i][0] = data[i][data[0].size() - 1];
	}
	// Разделим X и y на обучающие и тестовые данные:
	vector <vector <double>> X_train;
	X_train.resize(int(X.size() * 0.8), vector<double>(X[0].size()));
	vector <vector <double>> X_test;
	X_test.resize(X.size() - int(X.size() * 0.8), vector<double>(X[0].size()));

	vector <vector <double>> y_train;
	y_train.resize(int(y.size() * 0.8), vector<double>(y[0].size()));
	vector <vector <double>> y_test;
	y_test.resize(y.size() - int(y.size() * 0.8), vector<double>(y[0].size()));

	int k = 0;	// В цикле так же разделим данные на тестовые и тренировочные
	for (int i = 0; i < data.size(); i++) {	// Это нужно для того, чтобы справедливо оценить точность обученной сети
		if (i < int(X.size() * 0.8)) {		// Тестироваться сеть будет на тех данных, которые ей не встречались при обучении
			X_train[i] = X[i];
			y_train[i] = y[i];
		}
		else {
			k = i - X.size() * 0.8;
			X_test[k] = X[k];
			y_test[k] = y[k];
		}
	}
	vector <double> target;		// Объявим вспомогателньые вектора
	target.resize(neural.outlayer);
	vector<double> person;
	person.resize(neural.zerolayer);
	vector <double> predict = neural.predict(person);
	double miss = 0;			// miss - значение промаха, это то на сколько прогноз нейросети отличается от наших ожиданий
	double count = 0;			// count - количество раз, когда модель угадала диагноз 
	for (int i = 0; i < X_test.size(); i++) {
		person = X_test[i];
		target = y_test[i];
		predict = neural.predict(person);
		if (int(predict[0] + 0.5) == target[0]) {
			count++;
		}
	}
	double accuracy = count / X_test.size();	// Посчитаем точность, поделив count на количество рассмотренных в цикле пациентов
	count = 0;
	cout << "Accuracy before learning: " << accuracy << endl;
	// Теперь обучим нейросеть на тренировочных данных
	for (int i = 0; i < X_train.size(); i++) {
		person = X_train[i];
		target = y_train[i];
		predict = neural.predict(person);	// получаем предсказание
		miss = target[0] - predict[0];		// вычисляем промах
		neural.learn_back_prop(miss);		// запускаем алгоритм обратного распространения ошибки от полученного значения промаха
	}	// На этом обучение сети кончается, далее протестируем вычисленные значения весов связей
	for (int i = 0; i < X_test.size(); i++) {
		person = X_test[i];
		target = y_test[i];
		predict = neural.predict(person);	// Теперь произведём валидацию сети - посчитаем точность на ранее неизвестных сети данных
		miss = target[0] - predict[0];
		if (int(predict[0] + 0.5) == target[0]) {
			count++;
		}
	}
	accuracy = count / X_test.size();		// Вычисляем конечную точность
	cout << "Final accuracy:           " << accuracy << endl;
	cout << "Test predictions: ";
	cout << neural.predict(X[55])[0] << " - " << y[55][0] << endl;
	cout << neural.predict(X[601])[0] << " - " << y[601][0] << endl;
	return 0;
}
