// https://www.shadertoy.com/view/tsfyzn
// credits to @spalmer for grid smooth function https://www.shadertoy.com/view/wl3Sz2
#define MAX_STEPS 156
#define MIN_DISTANCE 0.01
#define MAX_DISTANCE 16.
#define GRID_SIZE 4.
#define speed 6.
#define MOUNTAIN_COLOR vec3(0.54, 0.11, 1.)
#define COLOR_PURPLE vec3(0.81, 0.19, 0.78)
#define COLOR_LIGHT vec3(0.14, 0.91, 0.98)
#define COLOR_SUN vec3(1., 0.56, 0.098)
#define MATERIAL_PLANE 1.
#define MATERIAL_BACK 2.
#define GRID_THICKNESS .2
#define COLOR_NIGHT_GRID vec3(0., .15, 0.)
#define COLOR_NIGHT_SUN vec3(0.5, .0, 0.)
#define COLOR_NIGHT_MOUNTAIN vec3(0.9, .3, 0.1)
#define SUNSET_SPEED 3.

vec3 lightPos = vec3(0., 3., -10.);

struct traceResult {
    bool  isHit;
    float distanceTo;
    float material;
    float planeHeight;
    vec3 planeNormal;
};

struct getDistResult {
    float distanceTo;
    float material;
    float planeHeight;
    vec3 planeNormal;
};

float sdPlane(vec3 p, float h) {
    return p.y - h;
}

float N21(vec2 p) {
    return fract(sin(p.x*223.32+p.y*5677.)*4332.23);
}

mat2 rot2d(float a) {
    float c = cos(a);
    float s = sin(a);
    return mat2(vec2(c,-s), vec2(s,c));
}

float getHeight(vec2 id) {
    //return 0.;
    float ax = abs(id.x);
    if (ax < GRID_SIZE) {
        return 0.;
    }

    float n = N21(id);

    float wave = sin(id.y/9. + cos(id.x/3.))*sin(id.x/9. + sin(id.y/4.));

    wave = clamp((wave * .5 + .5) + n*.15 - .6, 0., 1.);
    if (ax < (GRID_SIZE + 5.) && ax >= GRID_SIZE) {
        wave *= (ax - GRID_SIZE + 1.)*.2;
    }
    return (wave*10.);
}


getDistResult getDist(vec3 p) {
    float size = GRID_SIZE;
    vec3 nuv = p * size + vec3(0., 0., iTime * speed);
    vec2 uv = fract(nuv).xz;
    vec2 id = floor(nuv).xz;

    vec2 lv = uv;

    float bl = getHeight(id);
    float br = getHeight(id + vec2(1., 0.));
    float b = mix(bl, br, lv.x);

    float tl = getHeight(id + vec2(0., 1.));
    float tr = getHeight(id + vec2(1., 1.));
    float t = mix(tl, tr, lv.x);

    float height = mix(b,t, lv.y);

    float O = bl;
    float R = br;
    float T = getHeight(id + vec2(0. -1.));
    float B = tl;
    float L = getHeight(id + vec2(-1., 0));

    vec3 n = vec3(2.*(R-L), 2.*(B-T), -4.);


    float d = sdPlane(p, -.5 + 0.3*height);

    float db = -p.z + MAX_DISTANCE*.4;
    d = min(d, db);

    getDistResult result;

    result.distanceTo = d;
    result.material = MATERIAL_PLANE;
    result.planeHeight = height;
    result.planeNormal = normalize(n);

    if (d == db) {
        result.material = MATERIAL_BACK;
    }

    return result;
}

traceResult trace(vec3 ro, vec3 rd) {
    traceResult result;
    float ds, dt;
    getDistResult dist;
    for(int i = 0 ; i < MAX_STEPS ; i++) {
        vec3 p = ro + rd * ds;
        dist = getDist(p);
        dt = dist.distanceTo;
        ds += dt * .6;
        if (abs(dt) < MIN_DISTANCE || ds > MAX_DISTANCE) {
            break;
        }
    }
    result.isHit = abs(dt) < MIN_DISTANCE;
    result.distanceTo = ds;
    result.material = dist.material;
    result.planeHeight = dist.planeHeight;
    result.planeNormal = dist.planeNormal;
    return result;
}

float getLightDiffuse(vec3 p, float material, float height, vec3 normal) {
    vec3 l = normalize(lightPos - p);
    float dif = clamp(dot(normal, l), 0., 1.);
    return dif;
}

vec3 starsLayer(vec2 ouv) {
    vec3 col = vec3(0.);

    vec2 uv = fract(ouv) - .5;

    float d;

    for(int x = -1 ; x <= 1; x++) {
        for(int y = -1 ; y <= 1; y++) {
            vec2 offset = vec2(x,y);
            vec2 id = floor(ouv) + offset;
            float n = N21(id);
            if (n > .6) {
                float n1 = fract(n*123.432);
                float n2 = fract(n*1234.2432);

                float size = .01 + 0.05 * (n1 - .5);

                vec2 shift = vec2(n1 - .5, n2 - .5);
                d = max(d, size/length(uv - offset + shift));
            }
        }
    }


    return col + d*vec3(.1, .9, .1);
}

vec3 backgroundStars(vec2 uv) {
    vec3 col = vec3(0.);

    float t = iTime * (speed / 30.);

    float layers = 3.;

    for(float i = 0. ; i < 1. ; i+= 1./layers) {
        float depth = fract(i + t);
        float scale = mix(20., .5, depth);
        float fade = depth * smoothstep(1., .9, depth);

        col += starsLayer(uv * scale + i * 456.32) * fade;
    }
    return col;
}

vec3 getOthersideBackground(vec2 uv) {
    return backgroundStars(uv/8. + sin(iTime/(speed)));
}

vec3 getBackground(vec2 uv) {
    float set = 0. - clamp(sin(iTime/SUNSET_SPEED)*3., -1., 2.);

    float sunDist = length(uv + vec2(0., -2.5 - set));
    float sun = 1. - smoothstep(2.35, 2.5, sunDist);

    float gradient = sin(uv.y/4. - 3.14/32. + set/3.)*2.;
    float bands = abs(sin(uv.y * 8. + iTime*2.)) * (1. - step(2.5 + set, uv.y));

    float skyTop = 2.12/distance(uv, vec2(uv.x, 9.5));
    float skyBottom = 1.12/distance(uv, vec2(uv.x, -1.5));

    vec3 result;

    // sun

    if (set < -1.8) {
        result = vec3(sun) * (bands > 0. ? bands : 1.) * mix(vec3(0.), COLOR_NIGHT_SUN, ((abs(set) - 1.6) -.2) * 15.);
        float glow = smoothstep(.1, .5, (1.1)/sunDist);
        result += glow * COLOR_NIGHT_SUN;
    } else {
        result = vec3(sun * gradient * (bands > 0. ? bands : 1.)) * COLOR_SUN;
        //glow
        float glow = smoothstep(.1, .5, (1.1)/sunDist) + clamp(-1., 1., set);
        // result += glow * COLOR_PURPLE;

        // sky
        result += max(glow * COLOR_PURPLE, ((skyTop * MOUNTAIN_COLOR) + (skyBottom * COLOR_PURPLE))*(1. + set));
    }



    if (sun < .5) {
        // stars
        vec2 nuv = uv*2.;// + vec2(iTime, 0.);
        vec2 rize = vec2(-10., 12.);
        nuv -= rize;
        nuv *= rot2d(mod(-iTime/15., 6.28));
        nuv += rize;
        uv = fract(nuv);
        vec2 id = floor(nuv);
        uv -= .5;

        float n = N21(id);
        uv.x += fract(n*100.32) - .5;
        uv.y += fract(n*11323.432) - .5;

        float star = smoothstep(.5, 1., (0.03 + (0.02 * (fract(n*353.32) - .5)))/length(uv));

        result += star * step(.8, n);
    }

    return result;
}
float filterWidth2(vec2 uv)
{
     vec2 dx = dFdx(uv), dy = dFdy(uv);
    return dot(dx, dx) + dot(dy, dy) + .0001;
}

// (c) spalmer https://www.shadertoy.com/view/wl3Sz2
float gridPow(vec2 uv)
{
    vec2 p = uv * GRID_SIZE + vec2(0., iTime * speed);
    const float fadePower = 16.;
    vec2 f = fract(p);
    f = .5 - abs(.5 - f);
    f = max(vec2(0), 1. - f + .5*GRID_THICKNESS);
    f = pow(f, vec2(fadePower));
    float g = f.x+f.y; //max(f.x, f.y); //
    float s = sqrt(GRID_THICKNESS);
    return mix(g, s, exp2(-.01 / filterWidth2(p)));
}

vec3 getAlbedo(vec3 p, float material, float height, vec3 normal) {
    if (material == MATERIAL_BACK) {
        return getBackground(p.xy);
    }

    float sunSet = sin(iTime/SUNSET_SPEED)*.5 + .5;

    vec3 col = vec3(0.);
    float grid = gridPow(p.xz);

    float maxHeight = 2.5;

    vec3 grid_color = COLOR_PURPLE;
    vec3 cell_color = vec3(0.);
    vec3 mountain_color = MOUNTAIN_COLOR;
    mountain_color = mix(mountain_color, COLOR_NIGHT_MOUNTAIN, sunSet);



    if (height > 0.) {
        grid_color = mix(COLOR_PURPLE, COLOR_LIGHT, height/maxHeight);
        cell_color = mountain_color * mix(vec3(0.), mountain_color, height/maxHeight);
    }

    grid_color = mix(grid_color, COLOR_NIGHT_GRID, sunSet);

    col = mix(vec3(0.), grid_color, grid) + cell_color;

    return vec3(col);
}

float polarTriangle(vec2 uv, float offset) {
    float a = atan(uv.x, uv.y) + offset;
    float b = 6.28 / 3.;
    float l = length(uv);

    float d = cos(a - floor(.5 + a/b) * b) * l;

    return d;
}

float triangleMask(vec2 uv) {
    return polarTriangle(uv + vec2(0., -.1),3.14 + .5*sin(iTime));
}


void mainImage(out vec4 fragColor, in vec2 fragCoords) {
    vec2 uv = fragCoords.xy / iResolution.xy;
    uv -= .5;
    uv.x *= iResolution.x / iResolution.y;

    vec2 mouse = iMouse.xy / iResolution.xy;

    // lightPos.z = sin(iTime/3.)*100.;

    mouse.x = 0.5;
    mouse.y = 0.;

    vec3 col = vec3(0.);

    vec3 ro = vec3(0., .5, -.4);
    vec3 lookat = vec3(mouse.x*2.-1., 1. - mouse.y - .6, 0.);
    float zoom = .4;


    vec3 f = normalize(lookat - ro);
    vec3 r = normalize(cross(vec3(0., 1., 0), f));
    vec3 u = cross(f, r);
    vec3 c = ro + f * zoom;
    vec3 i = c + uv.x * r + uv.y * u;

    vec3 rd = normalize(i - ro);

    vec3 p = vec3(0.);

    traceResult tr = trace(ro, rd);

    if (tr.isHit) {

        p = ro + rd * tr.distanceTo;

        vec3 albedo = getAlbedo(p, tr.material, tr.planeHeight, tr.planeNormal);

        float diffuse = getLightDiffuse(p, tr.material, tr.planeHeight, tr.planeNormal);

        float fade = 1.;// - clamp((p.z-ro.z)/(MAX_DISTANCE * .8), 0., 1.);

        if (tr.material == MATERIAL_BACK) {
            col = albedo;
        } else {
            col = diffuse * albedo * fade;
        }

        float triangle = triangleMask(uv);
        float fd = fract(triangle - clamp(sin(iTime/3.), 0., 2.));
        float bc = (1. - step(.2, fd));

        col *= (tr.material == MATERIAL_BACK) ? bc : 1.;
        if (bc == 0.) {
            if (tr.material == MATERIAL_BACK) {
                col = getOthersideBackground(p.xy);
            } else {
                col *= vec3(.8);
            }
        }

        col += ((1. - step(.2, fd)) - (1. - step(.19, fd)))*.3;

    }


    fragColor = vec4(col, 1.);
    // fragColor.rgb = pow(fragColor.rgb, vec3(1.0/2.2));
}
