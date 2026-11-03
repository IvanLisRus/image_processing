#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstdint>

// Макросы для корректного экспорта функций в Windows (DLL) и Linux (SO)
#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif


// Изменяем размер изображения для передачи его в ИНС
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


// Возвращяет круглый кроп монеты, вокруг монеты заливаем цветом
extern "C" EXPORT void crop_coin_planar(
    const uint8_t* in_r,          // Входной канал R
    const uint8_t* in_g,          // Входной канал G
    const uint8_t* in_b,          // Входной канал B
    int32_t orig_w,               // Исходная ширина
    int32_t orig_h,               // Исходная высота
    float center_x,               // Нормированный X центра [0.0, 1.0]
    float center_y,               // Нормированный Y центра [0.0, 1.0]
    float width,                  // Нормированная ширина [0.0, 1.0]
    float height,                 // Нормированная высота [0.0, 1.0]
    uint8_t pad_r,                // Цвет фона: Red
    uint8_t pad_g,                // Цвет фона: Green
    uint8_t pad_b,                // Цвет фона: Blue
    uint8_t* out_r,               // Выходной канал R (буфер размером orig_w * orig_h)
    uint8_t* out_g,               // Выходной канал G (буфер размером orig_w * orig_h)
    uint8_t* out_b,               // Выходной канал B (буфер размером orig_w * orig_h)
    int32_t* crop_w,              // Выход: реальная ширина кропа
    int32_t* crop_h,              // Выход: реальная высота кропа
    int32_t* is_fully_visible     // Выход: 1 = монета целиком, 0 = обрезана
) {
    // 1. Базовая защита
    if (!in_r || !in_g || !in_b || !out_r || !out_g || !out_b || !crop_w || !crop_h || !is_fully_visible) {
        return;
    }
    if (orig_w <= 0 || orig_h <= 0) {
        *crop_w = 0;
        *crop_h = 0;
        *is_fully_visible = 0;
        return;
    }

    // 2. Денормализация координат и размеров
    float cx = center_x * orig_w;
    float cy = center_y * orig_h;
    float w_px = width * orig_w;
    float h_px = height * orig_h;

    // 3. Вычисление радиуса (диаметр = MAX(ширина, высота))
    float diameter = std::max(w_px, h_px);
    float radius = diameter / 2.0f;

    // 4. Вычисляем bounding box монеты (квадрат вокруг круга)
    int left = static_cast<int>(std::round(cx - radius));
    int top = static_cast<int>(std::round(cy - radius));
    int right = static_cast<int>(std::round(cx + radius));
    int bottom = static_cast<int>(std::round(cy + radius));

    // 5. Проверяем, полностью ли монета в кадре
    bool fully_visible = (left >= 0) && (right < orig_w) && (top >= 0) && (bottom < orig_h);
    *is_fully_visible = fully_visible ? 1 : 0;

    // 6. Обрезаем bounding box по границам изображения (если монета частично за краем)
    int crop_left = std::max(0, left);
    int crop_top = std::max(0, top);
    int crop_right = std::min(orig_w - 1, right);
    int crop_bottom = std::min(orig_h - 1, bottom);

    int final_width = crop_right - crop_left + 1;
    int final_height = crop_bottom - crop_top + 1;

    // Если кроп получился невалидным (монета вообще не в кадре)
    if (final_width <= 0 || final_height <= 0) {
        *crop_w = 0;
        *crop_h = 0;
        return;
    }

    // 7. Возвращаем реальные размеры кропа
    *crop_w = final_width;
    *crop_h = final_height;

    // 8. Создаем заголовки cv::Mat ПОВЕРХ массивов LabVIEW
    cv::Mat src_r_mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_r));
    cv::Mat src_g_mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_g));
    cv::Mat src_b_mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_b));

    cv::Mat dst_r_mat(orig_h, orig_w, CV_8UC1, out_r);
    cv::Mat dst_g_mat(orig_h, orig_w, CV_8UC1, out_g);
    cv::Mat dst_b_mat(orig_h, orig_w, CV_8UC1, out_b);

    // 9. Заполняем ВСЕ выходное изображение цветом фона
    dst_r_mat.setTo(pad_r);
    dst_g_mat.setTo(pad_g);
    dst_b_mat.setTo(pad_b);

    // 10. Определяем ROI для копирования (обрезанный bounding box)
    cv::Rect roi(crop_left, crop_top, final_width, final_height);

    // 11. Копируем область из исходного изображения в выходное (с учетом ROI)
    src_r_mat(roi).copyTo(dst_r_mat(roi));
    src_g_mat(roi).copyTo(dst_g_mat(roi));
    src_b_mat(roi).copyTo(dst_b_mat(roi));

    // 12. Создаем круговую маску и применяем её (затираем углы цветом фона)
    cv::Mat mask(final_height, final_width, CV_8UC1, cv::Scalar(0));
    // Центр круга в координатах ROI
    cv::Point circle_center(crop_left - left + static_cast<int>(std::round(radius)), 
                            crop_top - top + static_cast<int>(std::round(radius)));
    cv::circle(mask, circle_center, static_cast<int>(std::round(radius)), cv::Scalar(255), -1);
    
    // Применяем маску только к области кропа
    cv::Mat masked_r, masked_g, masked_b;
    dst_r_mat(roi).copyTo(masked_r, mask);
    dst_g_mat(roi).copyTo(masked_g, mask);
    dst_b_mat(roi).copyTo(masked_b, mask);
    
    masked_r.copyTo(dst_r_mat(roi));
    masked_g.copyTo(dst_g_mat(roi));
    masked_b.copyTo(dst_b_mat(roi));
}


// Поворот кропа монеты
extern "C" EXPORT void rotate_coin_planar(
    const uint8_t* in_r,          // Входной канал R
    const uint8_t* in_g,          // Входной канал G
    const uint8_t* in_b,          // Входной канал B
    int32_t orig_w,               // Исходная ширина
    int32_t orig_h,               // Исходная высота
    float angle,                  // Угол поворота в радианах
    uint8_t pad_r,                // Цвет фона: Red
    uint8_t pad_g,                // Цвет фона: Green
    uint8_t pad_b,                // Цвет фона: Blue
    uint8_t* out_r,               // Выходной канал R
    uint8_t* out_g,               // Выходной канал G
    uint8_t* out_b                // Выходной канал B
) {
    // 1. Базовая защита
    if (!in_r || !in_g || !in_b || !out_r || !out_g || !out_b) {
        return;
    }
    if (orig_w <= 0 || orig_h <= 0) {
        return;
    }

    // 2. Создаем заголовки cv::Mat ПОВЕРХ массивов LabVIEW
    cv::Mat src_r_mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_r));
    cv::Mat src_g_mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_g));
    cv::Mat src_b_mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_b));

    cv::Mat dst_r_mat(orig_h, orig_w, CV_8UC1, out_r);
    cv::Mat dst_g_mat(orig_h, orig_w, CV_8UC1, out_g);
    cv::Mat dst_b_mat(orig_h, orig_w, CV_8UC1, out_b);

    // 3. Вычисляем центр изображения
    cv::Point2f center(static_cast<float>(orig_w) / 2.0f, 
                       static_cast<float>(orig_h) / 2.0f);

    // 4. Преобразуем радианы в градусы (OpenCV работает в градусах)
    // Отрицательный угол для поворота против часовой стрелки
    double angle_degrees = angle * 180.0 / CV_PI;

    // 5. Создаем матрицу поворота
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle_degrees, 1.0);

    // 6. Настраиваем параметры для warpAffine
    // borderMode=BORDER_CONSTANT заполнит области за пределами изображения цветом pad
    cv::Scalar border_color_r(pad_r);
    cv::Scalar border_color_g(pad_g);
    cv::Scalar border_color_b(pad_b);

    // 7. Выполняем поворот для каждого канала
    cv::warpAffine(src_r_mat, dst_r_mat, rotation_matrix, 
                   cv::Size(orig_w, orig_h), 
                   cv::INTER_LINEAR, 
                   cv::BORDER_CONSTANT, 
                   border_color_r);

    cv::warpAffine(src_g_mat, dst_g_mat, rotation_matrix, 
                   cv::Size(orig_w, orig_h), 
                   cv::INTER_LINEAR, 
                   cv::BORDER_CONSTANT, 
                   border_color_g);

    cv::warpAffine(src_b_mat, dst_b_mat, rotation_matrix, 
                   cv::Size(orig_w, orig_h), 
                   cv::INTER_LINEAR, 
                   cv::BORDER_CONSTANT, 
                   border_color_b);
}




