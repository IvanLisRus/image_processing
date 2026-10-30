#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstdint>

// Макросы для корректного экспорта функций в Windows (DLL) и Linux (SO)
#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

extern "C" EXPORT void letterbox_resize_planar(
    const uint8_t* in_r,          // Входной канал R
    const uint8_t* in_g,          // Входной канал G
    const uint8_t* in_b,          // Входной канал B
    int32_t orig_w,               // Исходная ширина
    int32_t orig_h,               // Исходная высота
    int32_t target_w,             // Целевая ширина (например, 1024)
    int32_t target_h,             // Целевая высота (например, 1024 или 512)
    uint8_t pad_r,                // Цвет padding: Red
    uint8_t pad_g,                // Цвет padding: Green
    uint8_t pad_b,                // Цвет padding: Blue
    uint8_t* out_r,               // Выходной канал R (память выделена в LabVIEW)
    uint8_t* out_g,               // Выходной канал G
    uint8_t* out_b,               // Выходной канал B
    float* out_scale,             // Возвращаемый коэффициент масштабирования
    int32_t* out_left,            // Возвращаемый отступ слева
    int32_t* out_top              // Возвращаемый отступ сверху
) {
    // 1. Базовая защита от некорректных данных
    if (!in_r || !in_g || !in_b || !out_r || !out_g || !out_b || !out_scale || !out_left || !out_top) {
        return;
    }
    if (orig_w <= 0 || orig_h <= 0 || target_w <= 0 || target_h <= 0) {
        return;
    }

    // 2. Вычисляем масштаб и новые размеры
    float scale = std::min(static_cast<float>(target_w) / orig_w, 
                           static_cast<float>(target_h) / orig_h);
    
    // Гарантируем минимум 1 пиксель, чтобы избежать краха OpenCV при экстремальных пропорциях
    int new_w = std::max(1, static_cast<int>(orig_w * scale));
    int new_h = std::max(1, static_cast<int>(orig_h * scale));

    // 3. Вычисляем padding
    int pad_w = target_w - new_w;
    int pad_h = target_h - new_h;
    
    int top = pad_h / 2;
    int bottom = pad_h - top;
    int left = pad_w / 2;
    int right = pad_w - left;

    // 4. Создаем заголовки cv::Mat ПОВЕРХ существующих массивов LabVIEW (нулевое копирование)
    cv::Mat src_r_mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_r));
    cv::Mat src_g_mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_g));
    cv::Mat src_b_mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_b));

    cv::Mat dst_r_mat(target_h, target_w, CV_8UC1, out_r);
    cv::Mat dst_g_mat(target_h, target_w, CV_8UC1, out_g);
    cv::Mat dst_b_mat(target_h, target_w, CV_8UC1, out_b);

    // 5. Мгновенно заполняем все целевые изображения цветом padding
    dst_r_mat.setTo(pad_r);
    dst_g_mat.setTo(pad_g);
    dst_b_mat.setTo(pad_b);

    // 6. Определяем область интереса (ROI) для вставки масштабированного изображения
    cv::Rect roi(left, top, new_w, new_h);

    // 7. Масштабируем каждый канал напрямую в ROI целевого изображения
    cv::resize(src_r_mat, dst_r_mat(roi), cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::resize(src_g_mat, dst_g_mat(roi), cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::resize(src_b_mat, dst_b_mat(roi), cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 8. Возвращаем параметры для обратного преобразования координат
    *out_scale = scale;
    *out_left = left;
    *out_top = top;
}
