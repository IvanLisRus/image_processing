#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <vector>

// Макросы для корректного экспорта функций
#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif


// ============================================================================
// 1. LETTERBOX RESIZE (INTER_LINEAR)
// ============================================================================
extern "C" EXPORT void letterbox_resize_planar(
    const uint8_t* in_r, const uint8_t* in_g, const uint8_t* in_b,
    int32_t orig_w, int32_t orig_h,
    int32_t target_w, int32_t target_h,
    uint8_t pad_r, uint8_t pad_g, uint8_t pad_b,
    uint8_t* out_r, uint8_t* out_g, uint8_t* out_b,
    float* out_scale, int32_t* out_left, int32_t* out_top
) {
    if (!in_r || !in_g || !in_b || !out_r || !out_g || !out_b || !out_scale || !out_left || !out_top) {
        return;
    }
    if (orig_w <= 0 || orig_h <= 0 || target_w <= 0 || target_h <= 0) {
        return;
    }

    float scale = std::min(static_cast<float>(target_w) / orig_w,
                           static_cast<float>(target_h) / orig_h);

    int new_w = std::max(1, static_cast<int>(orig_w * scale));
    int new_h = std::max(1, static_cast<int>(orig_h * scale));

    int pad_w = target_w - new_w;
    int pad_h = target_h - new_h;

    int top = pad_h / 2;
    int left = pad_w / 2;

    cv::Mat src_r(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_r));
    cv::Mat src_g(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_g));
    cv::Mat src_b(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_b));

    cv::Mat dst_r(target_h, target_w, CV_8UC1, out_r);
    cv::Mat dst_g(target_h, target_w, CV_8UC1, out_g);
    cv::Mat dst_b(target_h, target_w, CV_8UC1, out_b);

    dst_r.setTo(pad_r);
    dst_g.setTo(pad_g);
    dst_b.setTo(pad_b);

    cv::Rect roi(left, top, new_w, new_h);

    cv::resize(src_r, dst_r(roi), cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::resize(src_g, dst_g(roi), cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::resize(src_b, dst_b(roi), cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    *out_scale = scale;
    *out_left = left;
    *out_top = top;
}


// ============================================================================
// 2. LETTERBOX RESIZE (INTER_CUBIC) — для совместимости с обучением
// ============================================================================
extern "C" EXPORT void letterbox_resize_cubic_planar(
    const uint8_t* in_r, const uint8_t* in_g, const uint8_t* in_b,
    int32_t orig_w, int32_t orig_h,
    int32_t target_w, int32_t target_h,
    uint8_t pad_r, uint8_t pad_g, uint8_t pad_b,
    uint8_t* out_r, uint8_t* out_g, uint8_t* out_b,
    float* out_scale, int32_t* out_left, int32_t* out_top
) {
    if (!in_r || !in_g || !in_b || !out_r || !out_g || !out_b || !out_scale || !out_left || !out_top) {
        return;
    }
    if (orig_w <= 0 || orig_h <= 0 || target_w <= 0 || target_h <= 0) {
        return;
    }

    float scale = std::min(static_cast<float>(target_w) / orig_w,
                           static_cast<float>(target_h) / orig_h);

    int new_w = std::max(1, static_cast<int>(orig_w * scale));
    int new_h = std::max(1, static_cast<int>(orig_h * scale));

    int pad_w = target_w - new_w;
    int pad_h = target_h - new_h;

    int top = pad_h / 2;
    int left = pad_w / 2;

    cv::Mat src_r(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_r));
    cv::Mat src_g(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_g));
    cv::Mat src_b(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_b));

    cv::Mat dst_r(target_h, target_w, CV_8UC1, out_r);
    cv::Mat dst_g(target_h, target_w, CV_8UC1, out_g);
    cv::Mat dst_b(target_h, target_w, CV_8UC1, out_b);

    dst_r.setTo(pad_r);
    dst_g.setTo(pad_g);
    dst_b.setTo(pad_b);

    cv::Rect roi(left, top, new_w, new_h);

    cv::resize(src_r, dst_r(roi), cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);
    cv::resize(src_g, dst_g(roi), cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);
    cv::resize(src_b, dst_b(roi), cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);

    *out_scale = scale;
    *out_left = left;
    *out_top = top;
}


// ============================================================================
// 3. SIMPLE RESIZE (INTER_CUBIC) — простой resize без letterbox
// ============================================================================
extern "C" EXPORT void resize_cubic_planar(
    const uint8_t* in_r, const uint8_t* in_g, const uint8_t* in_b,
    int32_t orig_w, int32_t orig_h,
    int32_t target_w, int32_t target_h,
    uint8_t* out_r, uint8_t* out_g, uint8_t* out_b
) {
    if (!in_r || !in_g || !in_b || !out_r || !out_g || !out_b) {
        return;
    }
    if (orig_w <= 0 || orig_h <= 0 || target_w <= 0 || target_h <= 0) {
        return;
    }

    cv::Mat src_r(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_r));
    cv::Mat src_g(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_g));
    cv::Mat src_b(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_b));

    cv::Mat dst_r(target_h, target_w, CV_8UC1, out_r);
    cv::Mat dst_g(target_h, target_w, CV_8UC1, out_g);
    cv::Mat dst_b(target_h, target_w, CV_8UC1, out_b);

    cv::resize(src_r, dst_r, cv::Size(target_w, target_h), 0, 0, cv::INTER_CUBIC);
    cv::resize(src_g, dst_g, cv::Size(target_w, target_h), 0, 0, cv::INTER_CUBIC);
    cv::resize(src_b, dst_b, cv::Size(target_w, target_h), 0, 0, cv::INTER_CUBIC);
}


// ============================================================================
// 4. CROP COIN — вырезание кропа монеты в форме СТРОГОГО КРУГА
// ============================================================================
extern "C" EXPORT void crop_coin_planar(
    const uint8_t* in_r, const uint8_t* in_g, const uint8_t* in_b,
    int32_t orig_w, int32_t orig_h,
    float center_x, float center_y, float width, float height,
    uint8_t pad_r, uint8_t pad_g, uint8_t pad_b,
    uint8_t* out_r, uint8_t* out_g, uint8_t* out_b,
    int32_t* crop_w, int32_t* crop_h,
    int32_t* is_fully_visible
) {
    if (!in_r || !in_g || !in_b || !out_r || !out_g || !out_b || !crop_w || !crop_h || !is_fully_visible) {
        return;
    }

    if (orig_w <= 0 || orig_h <= 0 || width <= 0.0f || height <= 0.0f) {
        *crop_w = 0;
        *crop_h = 0;
        *is_fully_visible = 0;
        return;
    }

    // 1. Денормализация координат центра монеты
    float cx = center_x * orig_w;
    float cy = center_y * orig_h;

    // 2. Расчет радиуса круга на основе максимальной стороны детекции
    float diameter_px = std::max(width * orig_w, height * orig_h);
    float radius = diameter_px / 2.0f;

    // 3. Вычисление включительных целочисленных границ [left, right] и [top, bottom].
    // Вычитание 1 из ceil переводит непрерывную координату в индекс последнего пикселя.
    // Эпсилон (1e-4f) защищает от погрешностей округления float (например, 1209.0001f -> 1210).
    constexpr float eps = 1e-4f;
    int left   = static_cast<int>(std::floor(cx - radius + eps));
    int top    = static_cast<int>(std::floor(cy - radius + eps));
    int right  = static_cast<int>(std::ceil(cx + radius - eps)) - 1;
    int bottom = static_cast<int>(std::ceil(cy + radius - eps)) - 1;

    // 4. Определение ПОЛНОЙ ВИДИМОСТИ круга в исходном кадре.
    bool fully_visible = (left >= 0) && (right < orig_w) && (top >= 0) && (bottom < orig_h);
    *is_fully_visible = fully_visible ? 1 : 0;

    // 5. Размеры выходного квадратного массива.
    // Жестко ограничиваем сторону меньшим измерением исходного изображения,
    // чтобы исключить переполнение выходного буфера при дробном сдвиге центра или width >= 1.0.
    int max_allowed_side = std::min(orig_w, orig_h);
    int side = std::max(right - left + 1, bottom - top + 1);

    if (side > max_allowed_side) {
        side = max_allowed_side;
        left = static_cast<int>(std::round(cx - side / 2.0f));
        top  = static_cast<int>(std::round(cy - side / 2.0f));
    }

    *crop_w = side;
    *crop_h = side;

    float radius_sq = radius * radius;

    // 6. Цикл по индексам ВЫХОДНОЙ КВАДРАТНОЙ матрицы
    for (int y = 0; y < side; ++y) {
        int src_y = top + y;
        bool y_inside = (src_y >= 0 && src_y < orig_h);
        size_t src_row_offset = y_inside ? (static_cast<size_t>(src_y) * orig_w) : 0;
        size_t dst_row_offset = static_cast<size_t>(y) * side;

        // Смещение на 0.5f переводит расчет расстояния в геометрический центр текущего пикселя
        float real_y = static_cast<float>(src_y) + 0.5f;
        float dy = real_y - cy;
        float dy_sq = dy * dy;

        for (int x = 0; x < side; ++x) {
            int src_x = left + x;
            size_t dst_idx = dst_row_offset + x;

            float real_x = static_cast<float>(src_x) + 0.5f;
            float dx = real_x - cx;

            // Проверяем попадание в строгий математический круг И физическое наличие пикселя в кадре
            if ((dx * dx + dy_sq) <= radius_sq && y_inside && (src_x >= 0 && src_x < orig_w)) {
                size_t src_idx = src_row_offset + src_x;
                out_r[dst_idx] = in_r[src_idx];
                out_g[dst_idx] = in_g[src_idx];
                out_b[dst_idx] = in_b[src_idx];
            } else {
                // Точка либо за пределами круга монеты, либо отрезана краем исходного изображения
                out_r[dst_idx] = pad_r;
                out_g[dst_idx] = pad_g;
                out_b[dst_idx] = pad_b;
            }
        }
    }
}


// ============================================================================
// 5. ROTATE COIN — поворот кропа монеты
// ============================================================================
extern "C" EXPORT void rotate_coin_planar(
    const uint8_t* in_r, const uint8_t* in_g, const uint8_t* in_b,
    int32_t orig_w, int32_t orig_h,
    float angle,
    uint8_t pad_r, uint8_t pad_g, uint8_t pad_b,
    uint8_t* out_r, uint8_t* out_g, uint8_t* out_b
) {
    if (!in_r || !in_g || !in_b || !out_r || !out_g || !out_b) {
        return;
    }
    if (orig_w <= 0 || orig_h <= 0) {
        return;
    }

    cv::Mat src_r(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_r));
    cv::Mat src_g(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_g));
    cv::Mat src_b(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_b));

    cv::Mat dst_r(orig_h, orig_w, CV_8UC1, out_r);
    cv::Mat dst_g(orig_h, orig_w, CV_8UC1, out_g);
    cv::Mat dst_b(orig_h, orig_w, CV_8UC1, out_b);

    cv::Point2f center(static_cast<float>(orig_w) / 2.0f,
                       static_cast<float>(orig_h) / 2.0f);

    double angle_degrees = angle * 180.0 / CV_PI;
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle_degrees, 1.0);

    cv::warpAffine(src_r, dst_r, rotation_matrix, cv::Size(orig_w, orig_h),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(pad_r));
    cv::warpAffine(src_g, dst_g, rotation_matrix, cv::Size(orig_w, orig_h),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(pad_g));
    cv::warpAffine(src_b, dst_b, rotation_matrix, cv::Size(orig_w, orig_h),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(pad_b));
}


// ============================================================================
// 6. GAMMA CORRECTION — ПРОДАКШН-ВЕРСИЯ
// ============================================================================
extern "C" EXPORT void apply_gamma_planar(
    const uint8_t* in_r, const uint8_t* in_g, const uint8_t* in_b,
    int32_t orig_w, int32_t orig_h,
    float gamma_value,
    uint8_t* out_r, uint8_t* out_g, uint8_t* out_b
) {
    // 1. Защита от некорректных параметров
    if (!in_r || !in_g || !in_b || !out_r || !out_g || !out_b) {
        return;
    }
    if (orig_w <= 0 || orig_h <= 0 || gamma_value <= 0.0f) {
        return;
    }

    // 2. Быстрый путь: если gamma == 1.0, просто копируем данные
    if (std::fabs(gamma_value - 1.0f) < 1e-5f) {
        size_t pixel_count = static_cast<size_t>(orig_w) * static_cast<size_t>(orig_h);
        std::memcpy(out_r, in_r, pixel_count);
        std::memcpy(out_g, in_g, pixel_count);
        std::memcpy(out_b, in_b, pixel_count);
        return;
    }

    // 3. Создаем заголовки cv::Mat поверх массивов LabVIEW (нулевое копирование)
    cv::Mat src_r(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_r));
    cv::Mat src_g(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_g));
    cv::Mat src_b(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_b));

    cv::Mat dst_r(orig_h, orig_w, CV_8UC1, out_r);
    cv::Mat dst_g(orig_h, orig_w, CV_8UC1, out_g);
    cv::Mat dst_b(orig_h, orig_w, CV_8UC1, out_b);

    // 4. Создаем LUT (Look-Up Table) для Gamma коррекции
    //    Формула идентична Python: table[i] = round((i/255)^invGamma * 255)
    float inv_gamma = 1.0f / gamma_value;
    std::vector<uint8_t> lut(256);
    for (int i = 0; i < 256; ++i) {
        float normalized = static_cast<float>(i) / 255.0f;
        float corrected = std::pow(normalized, inv_gamma);
        lut[i] = static_cast<uint8_t>(std::round(corrected * 255.0f));
    }

    cv::Mat lut_mat(1, 256, CV_8UC1, lut.data());

    // 5. Применяем LUT к каждому каналу (cv::LUT — высокооптимизированная функция OpenCV)
    cv::LUT(src_r, lut_mat, dst_r);
    cv::LUT(src_g, lut_mat, dst_g);
    cv::LUT(src_b, lut_mat, dst_b);
}


// ============================================================================
// 7. CLAHE — контрастное улучшение
// ============================================================================
extern "C" EXPORT void apply_clahe_planar(
    const uint8_t* in_r, const uint8_t* in_g, const uint8_t* in_b,
    int32_t orig_w, int32_t orig_h,
    float clip_limit,
    int32_t tile_grid_w, int32_t tile_grid_h,
    uint8_t* out_r, uint8_t* out_g, uint8_t* out_b
) {
    if (!in_r || !in_g || !in_b || !out_r || !out_g || !out_b) {
        return;
    }
    if (orig_w <= 0 || orig_h <= 0) {
        return;
    }

    cv::Mat in_mats[] = {
        cv::Mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_r)),
        cv::Mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_g)),
        cv::Mat(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_b))
    };

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

    cv::Mat out_mats[] = {
        cv::Mat(orig_h, orig_w, CV_8UC1, out_r),
        cv::Mat(orig_h, orig_w, CV_8UC1, out_g),
        cv::Mat(orig_h, orig_w, CV_8UC1, out_b)
    };
    cv::split(rgb_mat, out_mats);
}


// ============================================================================
// 8. GAMMA + CLAHE — комбинированная (максимально быстрая)
// ============================================================================
extern "C" EXPORT void apply_gamma_clahe_planar(
    const uint8_t* in_r, const uint8_t* in_g, const uint8_t* in_b,
    int32_t orig_w, int32_t orig_h,
    float gamma_value,
    float clip_limit,
    int32_t tile_grid_w, int32_t tile_grid_h,
    uint8_t* out_r, uint8_t* out_g, uint8_t* out_b
) {
    if (!in_r || !in_g || !in_b || !out_r || !out_g || !out_b) {
        return;
    }
    if (orig_w <= 0 || orig_h <= 0) {
        return;
    }

    cv::Mat src_r(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_r));
    cv::Mat src_g(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_g));
    cv::Mat src_b(orig_h, orig_w, CV_8UC1, const_cast<uint8_t*>(in_b));

    cv::Mat dst_r(orig_h, orig_w, CV_8UC1, out_r);
    cv::Mat dst_g(orig_h, orig_w, CV_8UC1, out_g);
    cv::Mat dst_b(orig_h, orig_w, CV_8UC1, out_b);

    // Шаг 1: Gamma коррекция
    if (gamma_value > 0.0f && std::fabs(gamma_value - 1.0f) >= 1e-5f) {
        float inv_gamma = 1.0f / gamma_value;
        std::vector<uint8_t> lut(256);
        for (int i = 0; i < 256; ++i) {
            float normalized = static_cast<float>(i) / 255.0f;
            float corrected = std::pow(normalized, inv_gamma);
            lut[i] = static_cast<uint8_t>(std::round(corrected * 255.0f));
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

    // Шаг 2: CLAHE
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

    cv::Mat out_mats[] = { dst_r, dst_g, dst_b };
    cv::split(rgb_mat, out_mats);
}
