#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/texture.h>

// std::cout << "!!!" << std::endl;

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class PhotonEmitter final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_medium, m_to_world)
    MI_IMPORT_TYPES(Scene, Texture)

    PhotonEmitter(const Properties &props) : Base(props) {
        /// The emitter lies at a single point in space
        m_flags = +EmitterFlags::DeltaPosition;
        // m_flags = +EmitterFlags::Empty;
        m_intensity = props.texture_d65<Texture>("intensity", 1.f);

        if (m_intensity->is_spatially_varying())
            Throw("The parameter 'intensity' cannot be spatially varying (e.g. bitmap type)!");

        dr::set_attr(this, "flags", m_flags);
        // degree to radiance: degree * pi / 180
        m_cutoff_angle = dr::deg_to_rad(0.01f);
        m_beam_width   = dr::deg_to_rad(0.01f*3.0f / 4.0f);
        // if the m_cutoff_angle is equal to m_beam_width, the denominator will be 0, it's impossible!
        m_inv_transition_width = 1.0f / (m_cutoff_angle - m_beam_width);
        
        m_cos_cutoff_angle = dr::cos(m_cutoff_angle);
        m_cos_beam_width   = dr::cos(m_beam_width);
        Assert(dr::all(m_cutoff_angle >= m_beam_width));
        // Avoid baking
        dr::make_opaque(m_beam_width, m_cutoff_angle,
                        m_cos_beam_width, m_cos_cutoff_angle,
                        m_inv_transition_width);
    }


    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &spatial_sample,
                                          const Point2f & /*dir_sample*/,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // 1. Sample directional component
        // Vector3f local_dir = warp::square_to_uniform_cone(spatial_sample, (Float) m_cos_cutoff_angle);
        Vector3f fixed_dir = Vector3f(0.f, 0.f, 1.f);
        Vector3f local_dir = dr::normalize(fixed_dir);
        // Float pdf_dir = warp::square_to_uniform_cone_pdf(local_dir, (Float) m_cos_cutoff_angle);
        
        // if (pdf_dir * Vector3f(0,0,1) != fixed_dir)
        //     pdf_dir = 0;
        Float pdf_dir = 445029;
                        
        // 2. Sample spectrum
        // defining a SurfaceInteraction3f object si and initializing its position, time and UV values
        auto si = dr::zeros<SurfaceInteraction3f>();
        si.time = time;
        si.p    = m_to_world.value().translation();
        si.uv   = Point2f(0.5,0.5);
        // generate a set of random wavelengths and the corresponding spectral weight
        auto [wavelengths, spec_weight] =
            sample_wavelengths(si, wavelength_sample, active);
        
        //  compute a falloff value for the given direction and active component.
        Float falloff = 1.0f;
        Ray3f result = Ray3f(si.p, m_to_world.value() * local_dir, time, wavelengths);
        // std::cout << result << std::endl;
        return { result, (spec_weight * falloff / pdf_dir) };
    } 

    std::pair<DirectionSample3f, Spectrum> sample_direction(const Interaction3f &it,
                                                            const Point2f &/*sample*/,
                                                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);
        DirectionSample3f ds;
        ds.p        = m_to_world.value().translation();
        ds.n        = 0.f;
        ds.uv       = 0.f;
        ds.pdf      = 1.f;
        ds.time     = it.time;
        ds.delta    = true; 
        ds.emitter  = this;
        ds.d        = ds.p - it.p;
        ds.dist     = dr::norm(ds.d);
        Float inv_dist = dr::rcp(ds.dist);
        ds.d        *= inv_dist;
        Vector3f local_d = m_to_world.value().inverse() * -ds.d;
        // Evaluate emitted radiance & falloff profile
        // Float falloff = falloff_curve(local_d, active);
        Float falloff = 1.0f;
        active &= falloff > 0.f;  // Avoid invalid texture lookups

        SurfaceInteraction3f si      = dr::zeros<SurfaceInteraction3f>();
        si.t                         = 0.f;
        si.time                      = it.time;
        si.wavelengths               = it.wavelengths;
        si.p                         = ds.p;
        UnpolarizedSpectrum radiance = m_intensity->eval(si, active);

        return { ds, depolarizer<Spectrum>(radiance & active) * (falloff * dr::sqr(inv_dist)) };
    }

    Float pdf_direction(const Interaction3f &,
                        const DirectionSample3f &, Mask) const override {
        return 0.f;
    }

    std::pair<PositionSample3f, Float>
    sample_position(Float time, const Point2f & /*sample*/,
                    Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSamplePosition, active);

        Vector3f center_dir = m_to_world.value() * ScalarVector3f(0.f, 0.f, 1.f);
        PositionSample3f ps(
            /* position */ m_to_world.value().translation(), center_dir,
            /*uv*/ Point2f(0.5f), time, /*pdf*/ 1.f, /*delta*/ true
        );
        return { ps, Float(1.f) };
    }

    std::pair<Wavelength, Spectrum>
    sample_wavelengths(const SurfaceInteraction3f &si, Float sample,
                       Mask active) const override {
        Wavelength wav;
        Spectrum weight;
        // std::tie(wav, weight) = m_intensity->sample_spectrum(
        //         si, math::sample_shifted<Wavelength>(sample), active);
        std::tie(wav, weight) = m_intensity->sample_spectrum(
                si, sample, active);

        return { wav, weight };
    }

    Spectrum eval(const SurfaceInteraction3f &, Mask) const override {
        return 0.f;
    }

    ScalarBoundingBox3f bbox() const override {
        ScalarPoint3f p = m_to_world.scalar() * ScalarPoint3f(0.f);
        return ScalarBoundingBox3f(p, p);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "PhotonEmitter[" << std::endl
            << "  to_world = " << string::indent(m_to_world) << "," << std::endl
            << "  intensity = " << m_intensity << "," << std::endl
            << "  cutoff_angle = " << m_cutoff_angle << "," << std::endl
            << "  beam_width = " << m_beam_width << "," << std::endl
            << "  medium = " << (m_medium ? string::indent(m_medium) : "")
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    ref<Texture> m_intensity;
    Float m_beam_width, m_cutoff_angle;
    Float m_cos_beam_width, m_cos_cutoff_angle, m_inv_transition_width;
};


MI_IMPLEMENT_CLASS_VARIANT(PhotonEmitter, Emitter)
MI_EXPORT_PLUGIN(PhotonEmitter, "Photon emitter")
NAMESPACE_END(mitsuba)
