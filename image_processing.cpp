#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <vector>

// Макросы для корректного экспорта функций в Windows (DLL) и Linux (SO)
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// ============================================================================
// 1. LETTERBOX RESIZE (INTER_LINEAR) - существующая функция
// ============================================================================
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

    // 7. Масштабируем каждый канал напрямую в ROI целевого изображения (INTER_LINEAR)
    cv::resize(src_r_mat, dst_r_mat(roi), cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::resize(src_g_mat, dst_g_mat(roi), cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::resize(src_b_mat, dst_b_mat(roi), cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 8. Возвращаем параметры для обратного преобразования координат
    *out_scale = scale;
    *out_left = left;
    *out_top = top;
}

// ============================================================================
// 2. LETTERBOX RESIZE (INTER_CUBIC) - НОВАЯ функция для совместимости с Python
// ============================================================================
extern "C" EXPORT void letterbox_resize_cubic_planar(
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
    
    // Гарантируем минимум 1 пиксель
    int new_w = std::max(1, static_cast<int>(orig_w * scale));
    int new_h = std::max(1, static_cast<int>(orig_h * scale));

    // 3. Вычисляем padding
    int pad_w = target_w - new_w;
    int pad_h = target_h - new_h;
    
    int top = pad_h / 2;
    int bottom = pad_h - top;
    int left = pad_w / 2;
    int right = pad_w - left;

    // 4. Создаем заголовки cv::Mat ПОВЕРХ массивов LabVIEW
    cv::Mat src_r_mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_r));
    cv::Mat src_g_mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_g));
    cv::Mat src_b_mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_b));

    cv::Mat dst_r_mat(target_h, target_w, CV_8UC1, out_r);
    cv::Mat dst_g_mat(target_h, target_w, CV_8UC1, out_g);
    cv::Mat dst_b_mat(target_h, target_w, CV_8UC1, out_b);

    // 5. Заполняем целевые изображения цветом padding
    dst_r_mat.setTo(pad_r);
    dst_g_mat.setTo(pad_g);
    dst_b_mat.setTo(pad_b);

    // 6. Определяем ROI
    cv::Rect roi(left, top, new_w, new_h);

    // 7. Масштабируем с INTER_CUBIC (как в Python train_angle_detector.py)
    cv::resize(src_r_mat, dst_r_mat(roi), cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);
    cv::resize(src_g_mat, dst_g_mat(roi), cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);
    cv::resize(src_b_mat, dst_b_mat(roi), cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);

    // 8. Возвращаем параметры
    *out_scale = scale;
    *out_left = left;
    *out_top = top;
}

// ============================================================================
// 3. SIMPLE RESIZE (INTER_CUBIC) - простой resize без letterbox
// ============================================================================
extern "C" EXPORT void resize_cubic_planar(
    const uint8_t* in_r,          // Входной канал R
    const uint8_t* in_g,          // Входной канал G
    const uint8_t* in_b,          // Входной канал B
    int32_t orig_w,               // Исходная ширина
    int32_t orig_h,               // Исходная высота
    int32_t target_w,             // Целевая ширина
    int32_t target_h,             // Целевая высота
    uint8_t* out_r,               // Выходной канал R
    uint8_t* out_g,               // Выходной канал G
    uint8_t* out_b                // Выходной канал B
) {
    if (!in_r || !in_g || !in_b || !out_r || !out_g || !out_b) {
        return;
    }
    if (orig_w <= 0 || orig_h <= 0 || target_w <= 0 || target_h <= 0) {
        return;
    }

    cv::Mat src_r_mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_r));
    cv::Mat src_g_mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_g));
    cv::Mat src_b_mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_b));

    cv::Mat dst_r_mat(target_h, target_w, CV_8UC1, out_r);
    cv::Mat dst_g_mat(target_h, target_w, CV_8UC1, out_g);
    cv::Mat dst_b_mat(target_h, target_w, CV_8UC1, out_b);

    // Resize с INTER_CUBIC
    cv::resize(src_r_mat, dst_r_mat, cv::Size(target_w, target_h), 0, 0, cv::INTER_CUBIC);
    cv::resize(src_g_mat, dst_g_mat, cv::Size(target_w, target_h), 0, 0, cv::INTER_CUBIC);
    cv::resize(src_b_mat, dst_b_mat, cv::Size(target_w, target_h), 0, 0, cv::INTER_CUBIC);
}

// ============================================================================
// 4. Gamma коррекция
// ============================================================================
extern "C" EXPORT void apply_gamma_planar(
    const uint8_t* in_r, const uint8_t* in_g, const uint8_t* in_b,
    int32_t orig_w, int32_t orig_h,
    float gamma_value,
    uint8_t* out_r, uint8_t* out_g, uint8_t* out_b
) {
    // 1. Базовая защита
    if (!in_r || !in_g || !in_b || !out_r || !out_g || !out_b) {
        return;
    }
    if (orig_w <= 0 || orig_h <= 0 || gamma_value <= 0.0f) {
        return;
    }

    // 2. Если gamma == 1.0, просто копируем данные
    if (std::fabs(gamma_value - 1.0f) < 1e-5f) {
        size_t pixel_count = static_cast<size_t>(orig_w) * static_cast<size_t>(orig_h);
        std::memcpy(out_r, in_r, pixel_count);
        std::memcpy(out_g, in_g, pixel_count);
        std::memcpy(out_b, in_b, pixel_count);
        return;
    }

    // 3. Создаем заголовки cv::Mat поверх массивов LabVIEW
    cv::Mat src_r(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_r));
    cv::Mat src_g(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_g));
    cv::Mat src_b(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_b));
    cv::Mat dst_r(orig_h, orig_w, CV_8UC1, out_r);
    cv::Mat dst_g(orig_h, orig_w, CV_8UC1, out_g);
    cv::Mat dst_b(orig_h, orig_w, CV_8UC1, out_b);

    // 4. Создаем LUT для Gamma коррекции
    float inv_gamma = 1.0f / gamma_value;
    std::vector<uint8_t> lut(256);
    for (int i = 0; i < 256; ++i) {
        float normalized = static_cast<float>(i) / 255.0f;
        float corrected = std::pow(normalized, inv_gamma);
        lut[i] = static_cast<uint8_t>(std::round(corrected * 255.0f));
    }
    
    // 5. Применяем LUT к каждому каналу
    cv::Mat lut_mat(1, 256, CV_8UC1, lut.data());
    cv::LUT(src_r, lut_mat, dst_r);
    cv::LUT(src_g, lut_mat, dst_g);
    cv::LUT(src_b, lut_mat, dst_b);
}


// ============================================================================
// 5. CLAHE (Contrast Limited Adaptive Histogram Equalization)
// ============================================================================
extern "C" EXPORT void apply_clahe_planar(
    const uint8_t* in_r, const uint8_t* in_g, const uint8_t* in_b,
    int32_t orig_w, int32_t orig_h,
    float clip_limit,             // Например, 2.0
    int32_t tile_grid_w,          // Например, 8
    int32_t tile_grid_h,          // Например, 8
    uint8_t* out_r, uint8_t* out_g, uint8_t* out_b
) {
    if (!in_r || !in_g || !in_b || !out_r || !out_g || !out_b) return;
    if (orig_w <= 0 || orig_h <= 0) return;

    // 1. Создаем заголовки Mat поверх массивов LabVIEW
    cv::Mat in_mats[] = {
        cv::Mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_r)),
        cv::Mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_g)),
        cv::Mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_b))
    };
    cv::Mat out_mats[] = {
        cv::Mat(orig_h, orig_w, CV_8UC1, out_r),
        cv::Mat(orig_h, orig_w, CV_8UC1, out_g),
        cv::Mat(orig_h, orig_w, CV_8UC1, out_b)
    };

    // 2. Объединяем в RGB
    cv::Mat rgb_mat;
    cv::merge(in_mats, 3, rgb_mat);

    // 3. Конвертируем RGB -> LAB
    cv::Mat lab_mat;
    cv::cvtColor(rgb_mat, lab_mat, cv::COLOR_RGB2Lab);

    // 4. Разделяем на каналы L, A, B
    std::vector<cv::Mat> lab_channels;
    cv::split(lab_mat, lab_channels);

    // 5. Применяем CLAHE только к каналу яркости (L)
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_limit, cv::Size(tile_grid_w, tile_grid_h));
    clahe->apply(lab_channels[0], lab_channels[0]);

    // 6. Объединяем обратно и конвертируем LAB -> RGB
    cv::merge(lab_channels, lab_mat);
    cv::cvtColor(lab_mat, rgb_mat, cv::COLOR_Lab2RGB);

    // 7. Разделяем результат обратно в плоские массивы LabVIEW
    cv::split(rgb_mat, out_mats);
}

// ============================================================================
// 6. Gamma + CLAHE (Комбинированная, максимально быстрая)
// ============================================================================
extern "C" EXPORT void apply_gamma_clahe_planar(
    const uint8_t* in_r, const uint8_t* in_g, const uint8_t* in_b,
    int32_t orig_w, int32_t orig_h,
    float gamma_value,            // Например, 0.9
    float clip_limit,             // Например, 2.0
    int32_t tile_grid_w,          // Например, 8
    int32_t tile_grid_h,          // Например, 8
    uint8_t* out_r, uint8_t* out_g, uint8_t* out_b
) {
    if (!in_r || !in_g || !in_b || !out_r || !out_g || !out_b) return;
    if (orig_w <= 0 || orig_h <= 0) return;

    cv::Mat src_r(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_r));
    cv::Mat src_g(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_g));
    cv::Mat src_b(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_b));
    
    // Используем выходные буферы как временные для Gamma
    cv::Mat dst_r(orig_h, orig_w, CV_8UC1, out_r);
    cv::Mat dst_g(orig_h, orig_w, CV_8UC1, out_g);
    cv::Mat dst_b(orig_h, orig_w, CV_8UC1, out_b);

    // Шаг 1: Gamma коррекция (сразу пишем в out_*)
    if (gamma_value > 0.0f && std::abs(gamma_value - 1.0f) >= 1e-5f) {
        float inv_gamma = 1.0f / gamma_value;
        std::vector<uint8_t> lut(256);
        for (int i = 0; i < 256; ++i) {
            lut[i] = static_cast<uint8_t>(std::round(std::pow(i / 255.0f, inv_gamma) * 255.0f));
        }
        cv::Mat lut_mat(1, 256, CV_8UC1, lut.data());
        cv::LUT(src_r, lut_mat, dst_r);
        cv::LUT(src_g, lut_mat, dst_g);
        cv::LUT(src_b, lut_mat, dst_b);
    } else {
        dst_r = src_r.clone();
        dst_g = src_g.clone();
        dst_b = src_b.clone();
    }

    // Шаг 2: CLAHE (используем уже обработанные Gamma буферы)
    cv::Mat in_mats[] = { dst_r, dst_g, dst_b };
    cv::Mat rgb_mat;
    cv::merge(in_mats, 3, rgb_mat);

    cv::Mat lab_mat;
    cv::cvtColor(rgb_mat, lab_mat, cv::COLOR_RGB2Lab);

    std::vector<cv::Mat> lab_channels;
    cv::split(lab_mat, lab_channels);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_limit, cv::Size(tile_grid_w, tile_grid_h));
    clahe->apply(lab_channels[0], lab_channels[0]);

    cv::merge(lab_channels, lab_mat);
    cv::cvtColor(lab_mat, rgb_mat, cv::COLOR_Lab2RGB);

    // Шаг 3: Разделяем итоговый результат обратно в out_r, out_g, out_b
    cv::Mat out_mats[] = { dst_r, dst_g, dst_b };
    cv::split(rgb_mat, out_mats);
}
