// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <tuple>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(float x, float y, const Vector3f* _v)
{
    
    Vector3f point_a = _v[0];
    Vector3f point_b = _v[1];
    Vector3f point_c = _v[2];

    
    Vector3f vectorA = Vector3f(x - point_a.x(), y - point_a.y(), 0);
    Vector3f vectorB = Vector3f(x - point_b.x(), y - point_b.y(), 0);
    Vector3f vectorC = Vector3f(x - point_c.x(), y - point_c.y(), 0);

    
    Vector3f edge0 = point_b - point_a;
    Vector3f edge1 = point_c - point_b;
    Vector3f edge2 = point_a - point_c;
    
    auto crossA = edge0.cross(vectorA);
    auto crossB = edge1.cross(vectorB);
    auto crossC = edge2.cross(vectorC);
    /*
        edge.cross(vector)实现三维向量的叉乘运算;
        判断所有叉乘的大小，同正或同负表示点位于三角形所有边的同一侧，点在三角形内部
    */
    if (crossA.z() > 0 && crossB.z() > 0 && crossC.z() > 0)return true;
    else if (crossA.z() < 0 && crossB.z() < 0 && crossC.z() < 0)return true;
    else return false;
}


static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}


void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division 
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation  
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        super_rasterize_triangle(t);
    }
}


void rst::rasterizer::super_rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    
    float minX, minY, maxX, maxY;
    minX = std::min(v[0].x(), std::min(v[1].x(), v[2].x()));
    maxX = std::max(v[0].x(), std::max(v[1].x(), v[2].x()));
    minY = std::min(v[0].y(), std::min(v[1].y(), v[2].y()));
    maxY = std::max(v[0].y(), std::max(v[1].y(), v[2].y()));

    for (int x = std::floor(minX); x <= std::ceil(maxX); x++) {
        for (int y = std::floor(minY); y <= std::ceil(maxY); y++) {
            
            int cnt = 0; 
            /* 
                cnt 记录有多少个采样点通过了深度测试;
                superOffsetX 和 superOffsetY 是预定义的数组，表示在每个像素内的 4 个采样点
            */
            for (int offset = 0; offset < 4; offset++) { 
                if (insideTriangle(x + superOffsetX[offset], y + superOffsetY[offset],t.v)){
                    float alpha, beta, gamma;
                    std::tie(alpha, beta, gamma) = computeBarycentric2D(x + superOffsetX[offset], y + superOffsetY[offset], t.v);
                    float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;
                
                    if (z_interpolated < super_depth_buf[get_super_index(x * 2 + offset % 2,y * 2 + offset / 2)]) {
                        cnt ++ ;
                        super_depth_buf[get_super_index(x * 2 + offset % 2,y * 2 + offset / 2)] = z_interpolated;
                        sample_color_buf[get_super_index(x * 2 + offset % 2,y * 2 + offset / 2)] = t.getColor();
                    }
                }
            }
            if(cnt > 0){
                Vector3f point = {(float)x, (float)y, 0};
                Vector3f color = {0, 0, 0};
                color += sample_color_buf[get_super_index(x * 2, y * 2)];
                color += sample_color_buf[get_super_index(x * 2 + 1, y * 2)];
                color += sample_color_buf[get_super_index(x * 2, y * 2 + 1)];
                color += sample_color_buf[get_super_index(x * 2 + 1, y * 2 + 1)];
                color /= 4.0f;
                set_pixel(point, color);
                /*
                    如果当前像素的 4 个采样点中有通过深度测试的，计算这些采样点的平均颜色;
                    最终调用 set_pixel() 函数，将像素 (x, y) 的颜色设置为 4 个采样点颜色的平均值
                */
            }
        }
    }
}


//Screen space rasterization 
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4(); //三角形的顶点转换为 4D 向量
    

    float minX, minY, maxX, maxY;
    minX = std::min(v[0].x(), std::min(v[1].x(), v[2].x()));
    maxX = std::max(v[0].x(), std::max(v[1].x(), v[2].x()));
    minY = std::min(v[0].y(), std::min(v[1].y(), v[2].y()));
    maxY = std::max(v[0].y(), std::max(v[1].y(), v[2].y()));
    /*
        计算三角形的包围盒，即三角形在屏幕上的最小和最大 x/y 值;
        通过 std::min 和 std::max 获取三角形的顶点坐标中最小和最大的 x 和 y 值，确定一个矩形框，包围整个三角形
    */
    for (int x = std::floor(minX); x <= std::ceil(maxX); x++) { //检查像素 (x, y) 是否在三角形内部
        for (int y = std::floor(minY); y <= std::ceil(maxY); y++) {
            if (insideTriangle(x + 0.5, y + 0.5, t.v)) {
                float alpha, beta, gamma;
                std::tie(alpha, beta, gamma) = computeBarycentric2D(x, y, t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w()); 
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                /*
                    w_reciprocal 是逆齐次坐标的 w 值，用于调整 z 值的插值;
                    由于顶点可能经过透视投影，所以插值 z 值时需要通过权重 alpha、beta、gamma 计算得到的值进行校正
                */
                z_interpolated *= w_reciprocal;

                int index = get_index(x, y);
                if (z_interpolated < depth_buf[index]) {
                    depth_buf[index] = z_interpolated;
                    auto color = t.getColor();
                    set_pixel(Eigen::Vector3f(x, y, z_interpolated), color);
                    /*
                        通过比较当前插值得到的 z_interpolated 与深度缓冲区中的值，决定是否需要更新该像素;
                        如果满足条件，更新深度缓冲区之后，会调用 set_pixel 函数，将当前像素 (x, y) 的颜色设置为三角形的颜色 t.getColor()，同时记录 z 值
                    */
                }
            }         
        }
    }
}


void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}


void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}


void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    /*
        检查并清除缓冲区；
        调用 std::fill 将frame_buf和sample_color_buf填充为 Eigen::Vector3f{0, 0, 0}，即黑色;
        用 std::numeric_limits<float>::infinity() 将depth_buf和super_depth_buf填充为无穷大 (infinity)，确保新渲染的像素可以通过深度测试
    */
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        std::fill(sample_color_buf.begin(), sample_color_buf.end(), Eigen::Vector3f{0, 0, 0});
        
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        std::fill(super_depth_buf.begin(), super_depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    super_depth_buf.resize(w * h * 4);
    sample_color_buf.resize(w * h * 4);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

int rst::rasterizer::get_super_index(int x, int y) {
    return (height * 2 - 1 - y) * width * 2 + x;
    /*
        (height * 2 - 1 - y)表示对y坐标进行翻转;
        ((height * 2 - 1 - y) * width * 2 + x)通过将 y 乘以每行的宽度，再加上 x 的偏移量，得到在一维数组中的位置索引
    */
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;
}



// clang-format on