#include "Plugin.h"

#include "BinaryData.h"
#include "Editor.h"

namespace Crunchypy
{

// ===========================================================================
// Helpers
// ===========================================================================

namespace
{

// RTNeural's xsimd backend reads a padded, RTNEURAL_DEFAULT_ALIGNMENT-aligned
// block of ceil(kInputSize / v_size) * v_size floats on every forward(). A bare
// std::array<float, kInputSize> would be both misaligned and read 3 floats out
// of bounds for 13 features, which the SIMD path sees as garbage/NaN. All
// inference input frames are therefore built in a zero-initialised, padded,
// aligned buffer; its unused SIMD tail lanes simply stay zero.
constexpr int kSimdLanes = RTNEURAL_DEFAULT_ALIGNMENT / sizeof (float);
constexpr int kPaddedInputSize = ((kInputSize + kSimdLanes - 1) / kSimdLanes) * kSimdLanes;
using AlignedFrame = std::array<float, kPaddedInputSize>;

static_assert (kPaddedInputSize >= kInputSize, "padded frame must cover all features");

/**
    Recomputes the embedded model.

    RTNeural's ModelT::parseJson returns void and reports every mismatch through
    debug_print alone. A model that does not match the template therefore loads as
    silent garbage rather than failing, which would still produce plausible
    looking audio. Everything below exists to make that failure loud instead: the
    JSON is validated against the compiled-in geometry first, and the network is
    then probed to confirm it actually holds trained weights.
*/
juce::String readEmbeddedModel (juce::String& errorOut)
{
    const auto* bytes = BinaryData::preamp_json;
    const auto size = static_cast<int> (BinaryData::preamp_jsonSize);

    if (bytes == nullptr || size <= 0)
    {
        errorOut = "Embedded model.json is missing or empty.";
        return {};
    }

    return juce::String::fromUTF8 (reinterpret_cast<const char*> (bytes), size);
}

/** Fails with a message describing the first structural mismatch found. */
bool validateGeometry (const nlohmann::json& j, juce::String& errorOut)
{
    if (! j.contains ("in_shape") || ! j["in_shape"].is_array())
    {
        errorOut = "model.json has no in_shape array.";
        return false;
    }

    const auto& inShape = j["in_shape"];
    if (inShape.back().get<int>() != kInputSize)
    {
        errorOut = juce::String ("model.json expects ") 
                       + juce::String (inShape.back().get<int>())
                       + " input features, this build is compiled for "
                       + juce::String (kInputSize) + ".";
        return false;
    }

    if (! j.contains ("layers") || ! j["layers"].is_array())
    {
        errorOut = "model.json has no layers array.";
        return false;
    }

    const auto& layers = j["layers"];

    // Three declared layers become four at runtime, because RTNeural expects the
    // tanh named by layer 0's "activation" as its own layer in the template.
    static const char* kExpectedTypes[] { "dense", "gru", "dense" };
    const auto expectedCount = static_cast<int> (sizeof (kExpectedTypes) / sizeof (kExpectedTypes[0]));

    if (static_cast<int> (layers.size()) != expectedCount)
    {
        errorOut = juce::String ("model.json declares ") + juce::String ((int) layers.size())
                       + " layers, this build expects " + juce::String (expectedCount) + ".";
        return false;
    }

    for (int i = 0; i < expectedCount; ++i)
    {
        const auto& l = layers.at ((size_t) i);

        if (! l.contains ("type") || l["type"].get<std::string>() != kExpectedTypes[i])
        {
            errorOut = juce::String ("model.json layer ") + juce::String (i) + " should be type "
                       + kExpectedTypes[i] + ".";
            return false;
        }

        // The middle layer is the only one whose width is the hidden size.
        if (i == 1 && l.contains ("shape") && l["shape"].is_array() && l["shape"].back().is_number())
        {
            const auto width = l["shape"].back().get<int>();
            if (width != kHiddenSize)
            {
                errorOut = juce::String ("model.json hidden size is ") + juce::String (width)
                           + ", this build is compiled for " + juce::String (kHiddenSize) + ".";
                return false;
            }
        }
    }

    // The tanh is the whole reason this build has four runtime layers. If it is
    // absent the network would be loaded as a different, wrong topology.
    const auto& first = layers.at (0);
    if (! first.contains ("activation") || first["activation"].get<std::string>() != "tanh")
    {
        errorOut = "model.json layer 0 is missing \"activation\":\"tanh\"; "
                   "the compiled network topology would not match.";
        return false;
    }

    return true;
}

/**
    Confirms the loaded network holds real weights.

    parseJson gives no failure signal, so a probe is the only way to tell a trained
    network from an unloaded one. A known input must produce a bounded, finite,
    non-zero response; an unloaded network produces exactly zero.

    Driven with a real signal rather than silence: a silent input is a degenerate
    probe, because a network has no reason to leave its zero-input fixed point.
*/
bool probeNetwork (Network& net, juce::String& errorOut)
{
    alignas (RTNEURAL_DEFAULT_ALIGNMENT) AlignedFrame frame {};
    frame[0] = 0.25f;

    // Fill the 12 knob slots (all of them, since they are all conditioned).
    // Use 0.5 for continuous knobs, 0 for switches to start from a known state.
    for (int k = 0; k < kNumKnobs; ++k)
        if (kFeatureSlot[(size_t) k] >= 0)
            frame[(size_t) kFeatureSlot[(size_t) k]] = 0.5f;

    net.reset();

    float peak = 0.0f;
    for (int i = 0; i < 64; ++i)
    {
        const auto y = net.forward (frame.data());
        peak = juce::jmax (peak, std::abs (y));
    }

    if (! std::isfinite (peak))
    {
        errorOut = "Loaded network produced a non-finite response; weights look corrupt.";
        return false;
    }

    if (peak < 1e-9f)
    {
        errorOut = "Loaded network produced silence, so its weights were not applied.";
        return false;
    }

    // Restore the zero state the plugin is expected to start from.
    net.reset();
    return true;
}

constexpr int kPowerampPaddedInputSize = ((kPowerampInputSize + kSimdLanes - 1) / kSimdLanes) * kSimdLanes;
using AlignedPowerampFrame = std::array<float, kPowerampPaddedInputSize>;

static_assert (kPowerampPaddedInputSize >= kPowerampInputSize, "padded poweramp frame must cover all features");

juce::String readEmbeddedPoweramp (juce::String& errorOut)
{
    const auto* bytes = BinaryData::poweramp_json;
    const auto size = static_cast<int> (BinaryData::poweramp_jsonSize);

    if (bytes == nullptr || size <= 0)
    {
        errorOut = "Embedded poweramp.json is missing or empty.";
        return {};
    }

    return juce::String::fromUTF8 (reinterpret_cast<const char*> (bytes), size);
}

bool validatePowerampGeometry (const nlohmann::json& j, juce::String& errorOut)
{
    if (! j.contains ("in_shape") || ! j["in_shape"].is_array())
    {
        errorOut = "poweramp.json has no in_shape array.";
        return false;
    }

    const auto& inShape = j["in_shape"];
    if (inShape.back().get<int>() != kPowerampInputSize)
    {
        errorOut = juce::String ("poweramp.json expects ")
                       + juce::String (inShape.back().get<int>())
                       + " input features, this build is compiled for "
                       + juce::String (kPowerampInputSize) + ".";
        return false;
    }

    if (! j.contains ("layers") || ! j["layers"].is_array())
    {
        errorOut = "poweramp.json has no layers array.";
        return false;
    }

    const auto& layers = j["layers"];
    static const char* kExpectedTypes[] { "dense", "gru", "dense" };
    const auto expectedCount = static_cast<int> (sizeof (kExpectedTypes) / sizeof (kExpectedTypes[0]));

    if (static_cast<int> (layers.size()) != expectedCount)
    {
        errorOut = juce::String ("poweramp.json declares ") + juce::String ((int) layers.size())
                       + " layers, this build expects " + juce::String (expectedCount) + ".";
        return false;
    }

    for (int i = 0; i < expectedCount; ++i)
    {
        const auto& l = layers.at ((size_t) i);
        if (! l.contains ("type") || l["type"].get<std::string>() != kExpectedTypes[i])
        {
            errorOut = juce::String ("poweramp.json layer ") + juce::String (i) + " should be type "
                       + kExpectedTypes[i] + ".";
            return false;
        }

        if (i == 1 && l.contains ("shape") && l["shape"].is_array() && l["shape"].back().is_number())
        {
            const auto width = l["shape"].back().get<int>();
            if (width != kPowerampHiddenSize)
            {
                errorOut = juce::String ("poweramp.json hidden size is ") + juce::String (width)
                           + ", this build is compiled for " + juce::String (kPowerampHiddenSize) + ".";
                return false;
            }
        }
    }

    const auto& first = layers.at (0);
    if (! first.contains ("activation") || first["activation"].get<std::string>() != "tanh")
    {
        errorOut = "poweramp.json layer 0 is missing \"activation\":\"tanh\"; "
                   "the compiled network topology would not match.";
        return false;
    }

    return true;
}

bool probePowerampNetwork (PowerampNetwork& net, juce::String& errorOut)
{
    alignas (RTNEURAL_DEFAULT_ALIGNMENT) AlignedPowerampFrame frame {};
    frame[0] = 0.25f; // audio signal
    frame[1] = 0.50f; // Presence knob

    net.reset();

    float peak = 0.0f;
    for (int i = 0; i < 64; ++i)
    {
        const auto y = net.forward (frame.data());
        peak = juce::jmax (peak, std::abs (y));
    }

    if (! std::isfinite (peak))
    {
        errorOut = "Loaded poweramp network produced a non-finite response; weights look corrupt.";
        return false;
    }

    if (peak < 1e-9f)
    {
        errorOut = "Loaded poweramp network produced silence, so its weights were not applied.";
        return false;
    }

    net.reset();
    return true;
}

} // namespace

// ===========================================================================
// Construction
// ===========================================================================

PluginProcessor::PluginProcessor()
    : AudioProcessor (BusesProperties()
                          .withInput ("Input", juce::AudioChannelSet::stereo(), true)
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      parameters (*this, nullptr, "MARKIICPLUS", makeLayout())
{
    // Initialise targets to match each parameter's declared default: continuous
    // knobs at 0.5, switches at 0 (OFF), Output/Input at 0 dB (unity), stage switches at 1 (ON).
    for (int i = 0; i < kNumParams; ++i)
    {
        if (i == kOutput)
            targets[(size_t) i].store (kOutputDefaultDb);
        else if (i == kInput)
            targets[(size_t) i].store (kInputDefaultDb);
        else if (i == kPreampSwitch || i == kPowerampSwitch || i == kEqSwitch)
            targets[(size_t) i].store (1.0f);
        else if (i == kStereoSwitch || i == kLimiterSwitch)
            targets[(size_t) i].store (0.0f);
        else if (i >= kFirstSwitchParam && i <= kLastSwitchParam)
            targets[(size_t) i].store (0.0f);
        else
            targets[(size_t) i].store (kDefaultKnob);
        parameters.addParameterListener (kParamId[i], this);
    }

    bypassValue = parameters.getRawParameterValue (kBypassParamId);

    loadModel();
    loadPowerampModel();
}

PluginProcessor::~PluginProcessor()
{
    for (int i = 0; i < kNumParams; ++i)
        parameters.removeParameterListener (kParamId[i], this);
}

juce::AudioProcessorValueTreeState::ParameterLayout PluginProcessor::makeLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    const auto range = juce::NormalisableRange<float> (0.0f, 1.0f);

    // 12 conditioned knobs, all 0..1, default 0.5. Parameter IDs are stable and
    // deliberately NOT in the panel's left-to-right order (that lives in the
    // editor's kKnobOrder) and NOT in the network's feature order (KNOB_ORDER).
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kVol1], 1 }, "Vol1", range, kDefaultKnob));
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kLeadDrive], 1 }, "Lead Drive", range, kDefaultKnob));
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kTreble], 1 }, "Treble", range, kDefaultKnob));
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kBass], 1 }, "Bass", range, kDefaultKnob));
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kMiddle], 1 }, "Middle", range, kDefaultKnob));
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kLeadMaster], 1 }, "Lead Master", range, kDefaultKnob));

    // Master 1 is a UI knob, not a conditioning channel: kFeatureSlot[kMaster1]
    // is -1. The editor applies it after the network as a voltage divide, so the
    // default here (0.5) is the position that reproduces the rendered reference.
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kMaster1], 1 }, "Master 1", range, kDefaultKnob));

    // 6 Pull switches, real booleans (the model's conditioning expects 0 or 1).
    // Default OFF; the editor toggles them when their knob is clicked. Unlike the
    // knobs they are written straight into the feature vector without smoothing,
    // so flipping one is instant change rather than a 15 ms slew.
    layout.add (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { kParamId[kPullBrightVol1], 1 }, "Pull Bright (Vol1)", false));
    layout.add (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { kParamId[kPullShiftTreble], 1 }, "Pull Shift (Treble)", false));
    layout.add (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { kParamId[kPullShiftBass], 1 }, "Pull Shift (Bass)", false));
    layout.add (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { kParamId[kPullShiftMod], 1 }, "Pull Shift (++ mod)", false));
    layout.add (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { kParamId[kPullDeepMaster1], 1 }, "Pull Deep (Master1)", false));
    layout.add (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { kParamId[kPullBrightLeadMaster], 1 }, "Pull Bright (Lead Master)", false));

    // Output level in dB, -24 to +6, default 0 dB (unity).
    const auto outRange = juce::NormalisableRange<float> (kOutputMinDb, kOutputMaxDb, kOutputStepDb);
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kOutput], 1 }, "Output", outRange, kOutputDefaultDb));

    // Input level in dB, -24 to +24, default 0 dB (unity).
    const auto inRange = juce::NormalisableRange<float> (kInputMinDb, kInputMaxDb, kInputStepDb);
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kInput], 1 }, "Input", inRange, kInputDefaultDb));

    // 5-band graphic EQ sliders, all 0..1, default 0.5 (the noon response).
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kEq80], 1 }, "EQ 80", range, kDefaultKnob));
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kEq240], 1 }, "EQ 240", range, kDefaultKnob));
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kEq750], 1 }, "EQ 750", range, kDefaultKnob));
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kEq2200], 1 }, "EQ 2200", range, kDefaultKnob));
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kEq6600], 1 }, "EQ 6600", range, kDefaultKnob));

    // Graphic EQ active switch (default true = EQ In)
    layout.add (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { kParamId[kEqSwitch], 1 }, "EQ Switch", true));

    // Poweramp Presence pot, 0..1, default 0.5 (nominal noon setting).
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kPresence], 1 }, "Presence", range, kDefaultKnob));

    // Preamp stage active switch (default true = Preamp In)
    layout.add (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { kParamId[kPreampSwitch], 1 }, "Preamp Switch", true));

    // Poweramp stage active switch (default true = Poweramp In)
    layout.add (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { kParamId[kPowerampSwitch], 1 }, "Poweramp Switch", true));

    // Processing mode: Stereo (true) vs Mono (false), default Mono (false)
    layout.add (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { kParamId[kStereoSwitch], 1 }, "Stereo", false));

    // Transparent safety limiter switch, default false (OFF)
    layout.add (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { kParamId[kLimiterSwitch], 1 }, "Limiter", false));

    // A real bypass, not a knob trick: hosts map it, and processBlockBypassed
    // passes audio through with every stage switched off.
    layout.add (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { kBypassParamId, 1 }, "Bypass", false));

    return layout;
}

// ===========================================================================
// Model loading
// ===========================================================================

bool PluginProcessor::loadModel()
{
    modelError.clear();
    modelKnobOrder.clear();
    exportValEsr = 0.0;

    const auto text = readEmbeddedModel (modelError);
    if (text.isEmpty())
        return false;

    // allow_exceptions = false, so malformed JSON returns a discarded value instead
    // of throwing. Throwing out of a constructor inside a host is a bad afternoon.
    const auto j = nlohmann::json::parse (text.toStdString(), nullptr, false);
    if (j.is_discarded())
    {
        modelError = "Embedded model.json is not valid JSON.";
        return false;
    }

    if (j.contains ("metadata"))
    {
        const auto& meta = j["metadata"];

        if (meta.contains ("knobs"))
            for (const auto& k : meta["knobs"])
                modelKnobOrder.add (juce::String (k.get<std::string>()));

        if (meta.contains ("val_esr"))
            exportValEsr = meta["val_esr"].get<double>();
    }

    if (! validateGeometry (j, modelError))
        return false;

    // The conditioning order is baked into the first layer's weights. If the exported
    // model ever changes its declared knob order, the cross-wiring is silent and the
    // failure shows up as "the knobs feel wrong", so it is checked here instead.
    //
    // The embedded model MUST declare exactly the 12 KNOB_ORDER knobs.
    if (modelKnobOrder.size() == (size_t) kNumKnobs)
    {
        for (int i = 0; i < kNumKnobs; ++i)
            if (modelKnobOrder[(size_t) i] != KNOB_ORDER[i])
            {
                modelError = juce::String ("model.json conditioning order is [")
                             + modelKnobOrder[0];
                for (int k = 1; k < kNumKnobs; ++k)
                    modelError += ", " + modelKnobOrder[k];
                modelError += "], this build is wired for [";
                modelError += KNOB_ORDER[0];
                for (int k = 1; k < kNumKnobs; ++k)
                    modelError += ", " + juce::String (KNOB_ORDER[k]);
                modelError += "].";
                return false;
            }
    }
    else
    {
        modelError = juce::String ("model.json declares ") + juce::String ((int) modelKnobOrder.size())
                       + " conditioned knobs, but this build expects " + juce::String (kNumKnobs) + ".";
        return false;
    }

    model[0].parseJson (j);
    model[1].parseJson (j);
    modelValid = probeNetwork (model[0], modelError);
    return modelValid;
}

bool PluginProcessor::loadPowerampModel()
{
    powerampError.clear();
    powerampValEsr = 0.0;

    const auto text = readEmbeddedPoweramp (powerampError);
    if (text.isEmpty())
        return false;

    const auto j = nlohmann::json::parse (text.toStdString(), nullptr, false);
    if (j.is_discarded())
    {
        powerampError = "Embedded poweramp.json is not valid JSON.";
        return false;
    }

    if (j.contains ("metadata"))
    {
        const auto& meta = j["metadata"];
        if (meta.contains ("val_esr"))
            powerampValEsr = meta["val_esr"].get<double>();
    }

    if (! validatePowerampGeometry (j, powerampError))
        return false;

    powerampModel[0].parseJson (j);
    powerampModel[1].parseJson (j);
    powerampValid = probePowerampNetwork (powerampModel[0], powerampError);
    return powerampValid;
}

void PluginProcessor::resetModelState()
{
    for (int ch = 0; ch < 2; ++ch)
    {
        if (modelValid)
            model[(size_t) ch].reset();
        if (powerampValid)
            powerampModel[(size_t) ch].reset();
    }
}

// ===========================================================================
// Parameters
// ===========================================================================

void PluginProcessor::parameterChanged (const juce::String& parameterID, float newValue)
{
    for (int i = 0; i < kNumParams; ++i)
        if (parameterID == kParamId[i])
        {
            targets[(size_t) i].store (newValue); // parameter value is already range-clamped by JUCE.
            return;
        }
}

// ===========================================================================
// Lifecycle
// ===========================================================================

void PluginProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    currentHostRate = sampleRate;
    needsResampling = std::abs (sampleRate - 48000.0) > 0.5;

    // Initialise smoothedParams from current target values.
    for (int i = 0; i < kNumParams; ++i)
        smoothedParams[(size_t) i] = targets[(size_t) i].load();

    // DC Blocker, EQ stage, and limiter run inside stepChannelSample at the model's native 48 kHz clock
    juce::dsp::ProcessSpec modelSpec;
    modelSpec.sampleRate = 48000.0;
    modelSpec.maximumBlockSize = static_cast<juce::uint32> (samplesPerBlock);
    modelSpec.numChannels = 1;

    for (int ch = 0; ch < 2; ++ch)
    {
        dcBlocker[(size_t) ch].prepare (modelSpec);
        dcBlocker[(size_t) ch].setType (juce::dsp::FirstOrderTPTFilterType::highpass);
        dcBlocker[(size_t) ch].setCutoffFrequency (15.0f);
        dcBlocker[(size_t) ch].reset();

        if (needsResampling)
            resampler[(size_t) ch].prepare (sampleRate);
        else
            resampler[(size_t) ch].release();

        limiter[(size_t) ch].prepare (48000.0);
        limiter[(size_t) ch].setCeiling (1.0f);

        eqStage[(size_t) ch].prepare (48000.0);
        eqStage[(size_t) ch].reset();
        eqStage[(size_t) ch].setEnabled (targets[(size_t) kEqSwitch].load() >= 0.5f);
        eqStage[(size_t) ch].setKnobs ({ targets[(size_t) kEq80].load(), targets[(size_t) kEq240].load(),
                                         targets[(size_t) kEq750].load(), targets[(size_t) kEq2200].load(),
                                         targets[(size_t) kEq6600].load() });

        model[(size_t) ch].reset();
        powerampModel[(size_t) ch].reset();
    }

    if (needsResampling)
        setLatencySamples (resampler[0].getLatencySamples());
    else
        setLatencySamples (0);
}

void PluginProcessor::releaseResources()
{
}

bool PluginProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& out = layouts.getMainOutputChannelSet();

    // Mono in, mono or stereo out. A preamp is a mono effect with a mono input,
    // so stereo input is only accepted to avoid dropping the user's signal.
    if (out != juce::AudioChannelSet::mono() && out != juce::AudioChannelSet::stereo())
        return false;

    const auto& in = layouts.getMainInputChannelSet();
    if (in != juce::AudioChannelSet::mono() && in != juce::AudioChannelSet::stereo())
        return false;

    return true;
}

double PluginProcessor::getTailLengthSeconds() const
{
    // The GRU rings on after the input stops, so there is a genuine tail. Reporting
    // zero makes hosts discard reverb tails and shorten offline renders.
    return 2.0;
}

// ===========================================================================
// Processing
//
// RTNeural's ModelT is a per-sample interface here: forward(const float*) takes one
// kInputSize vector and returns one output sample.
// The network is trained at 48000 Hz. If the host operates at 48000 Hz, audio flows
// directly with zero resampling and zero latency. At any other sample rate,
// band-limited resampling runs around the network (host rate -> 48 kHz ->
// network -> host rate) to ensure learned time constants and tone stack filters
// match the physical circuit exactly.
//
// Output chain: model (wet) -> Master 1 voltage divide -> Output gain in dB ->
// DC blocker -> soft-knee output limiter. Everything after the network runs at
// the host rate.
// ===========================================================================

juce::AudioProcessorParameter* PluginProcessor::getBypassParameter() const
{
    // Returning this makes the host's bypass button, our button and automation
    // one and the same switch.
    return parameters.getParameter (kBypassParamId);
}

void PluginProcessor::processBlockBypassed (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    // True bypass: channel 0 already holds the input, so only the extra output
    // channels need filling. No limiter, no DC blocker, no network state updates,
    // so an engaged bypass never disturbs the preamp's hidden state.
    const auto numSamples = buffer.getNumSamples();
    const auto numChannels = buffer.getNumChannels();

    if (numChannels > 1 && numSamples > 0)
    {
        const auto* in0 = buffer.getReadPointer (0);
        for (int ch = 1; ch < numChannels; ++ch)
            buffer.copyFrom (ch, 0, in0, numSamples);
    }
}

void PluginProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    const auto numSamples = buffer.getNumSamples();
    const auto numChannels = buffer.getNumChannels();

    if (numChannels == 0 || numSamples == 0)
        return;

    // Bypass is checked before anything else: even with a failed model embed, a
    // bypassed plugin is a cable, not a mute.
    if (bypassValue != nullptr && *bypassValue > 0.5f)
    {
        processBlockBypassed (buffer, midiMessages);
        return;
    }

    if (! modelValid)
    {
        // No network. Pass audio through untouched rather than emitting silence,
        // so a bad embed is audible as "no effect" instead of "broken host".
        buffer.clear();
        return;
    }

    // Block-rate parameter smoothing: time constant tau = 15 ms (0.015 s)
    // Ensures knobs and sliders react immediately (~30-50 ms to full settlement)
    // without zipper noise or multi-second lag regardless of host buffer size.
    const float blockSeconds = static_cast<float> (numSamples) / static_cast<float> (currentHostRate > 1000.0 ? currentHostRate : 48000.0);
    constexpr float kTauSeconds = 0.015f;
    const float blockSmoothingCoeff = static_cast<float> (1.0 - std::exp (-blockSeconds / kTauSeconds));

    // Update smoothed parameters
    for (int i = 0; i < kNumParams; ++i)
    {
        if ((i >= kFirstSwitchParam && i <= kLastSwitchParam)
            || i == kEqSwitch || i == kPreampSwitch || i == kPowerampSwitch || i == kStereoSwitch || i == kLimiterSwitch)
        {
            smoothedParams[(size_t) i] = targets[(size_t) i].load();
        }
        else
        {
            const float target = targets[(size_t) i].load();
            const float diff = target - smoothedParams[(size_t) i];
            if (std::abs (diff) < 1e-4f)
                smoothedParams[(size_t) i] = target;
            else
                smoothedParams[(size_t) i] += blockSmoothingCoeff * diff;
        }
    }

    const bool isStereoMode = (smoothedParams[(size_t) kStereoSwitch] > 0.5f) && (numChannels >= 2);
    const int activeProcessChannels = isStereoMode ? 2 : 1;

    alignas (RTNEURAL_DEFAULT_ALIGNMENT) AlignedFrame frame {};
    for (int i = 0; i < kNumParams; ++i)
    {
        if (kFeatureSlot[(size_t) i] >= 0)
            frame[(size_t) kFeatureSlot[(size_t) i]] = smoothedParams[(size_t) i];
    }

    const bool preampOn = smoothedParams[(size_t) kPreampSwitch] > 0.5f && modelValid;
    const bool eqOn = smoothedParams[(size_t) kEqSwitch] >= 0.5f;
    const bool powerampOn = smoothedParams[(size_t) kPowerampSwitch] > 0.5f && powerampValid;
    const bool limiterOn = smoothedParams[(size_t) kLimiterSwitch] > 0.5f;
    const float inGain = inputGainFromDb (smoothedParams[(size_t) kInput]);
    const float outGain = outputGainFromDb (smoothedParams[(size_t) kOutput]);
    const float m1Gain = master1GainFor (smoothedParams[(size_t) kMaster1]);

    for (int ch = 0; ch < 2; ++ch)
    {
        eqStage[(size_t) ch].setEnabled (eqOn);
        eqStage[(size_t) ch].setKnobs ({ smoothedParams[(size_t) kEq80], smoothedParams[(size_t) kEq240],
                                         smoothedParams[(size_t) kEq750], smoothedParams[(size_t) kEq2200],
                                         smoothedParams[(size_t) kEq6600] });
    }

    auto stepChannelSample = [this, &frame, preampOn, eqOn, powerampOn, limiterOn, inGain, outGain, m1Gain]
                             (int ch, float inSamp) noexcept -> float
    {
        // 1. Input trim
        const float x = inSamp * inGain;

        // 2. Preamp Stage
        float preampOut = x;
        if (preampOn)
        {
            frame[0] = x;
            preampOut = model[(size_t) ch].forward (frame.data());
        }

        // 3. Graphic EQ Stage
        const float eqOut = eqOn ? eqStage[(size_t) ch].process (preampOut) : preampOut;

        // 4. Master 1 voltage divide / poweramp drive
        const float powerampIn = eqOut * m1Gain;

        // 5. Poweramp Stage
        float powerampOut = powerampIn;
        if (powerampOn)
        {
            alignas (RTNEURAL_DEFAULT_ALIGNMENT) AlignedPowerampFrame paFrame {};
            paFrame[0] = powerampIn;
            paFrame[1] = smoothedParams[(size_t) kPresence];
            // Nominal level compensation (+1.14 dB) so poweramp engaged and bypassed match seamlessly in volume
            constexpr float kPowerampLevelTrim = 1.14f;
            powerampOut = powerampModel[(size_t) ch].forward (paFrame.data()) * kPowerampLevelTrim;
        }

        // 6. Output trim & optional transparent safety limiter
        const float trimmedOut = powerampOut * outGain;
        const float dcClean = dcBlocker[(size_t) ch].processSample (0, trimmedOut);
        return limiterOn ? limiter[(size_t) ch].processSample (dcClean) : dcClean;
    };

    if (! needsResampling)
    {
        if (! isStereoMode)
        {
            const auto* in0 = buffer.getReadPointer (0);
            const auto* in1 = numChannels > 1 ? buffer.getReadPointer (1) : nullptr;
            auto* out0 = buffer.getWritePointer (0);

            for (int n = 0; n < numSamples; ++n)
            {
                const float monoIn = in1 != nullptr ? 0.5f * (in0[n] + in1[n]) : in0[n];
                out0[n] = stepChannelSample (0, monoIn);
            }

            for (int ch = 1; ch < numChannels; ++ch)
                buffer.copyFrom (ch, 0, out0, numSamples);
        }
        else
        {
            auto* ch0 = buffer.getWritePointer (0);
            auto* ch1 = buffer.getWritePointer (1);

            for (int n = 0; n < numSamples; ++n)
            {
                ch0[n] = stepChannelSample (0, ch0[n]);
                ch1[n] = stepChannelSample (1, ch1[n]);
            }
        }
        return;
    }

    // Resampled path (host sample rate != 48 kHz)
    if (! isStereoMode)
    {
        auto* out0 = buffer.getWritePointer (0);
        if (numChannels > 1)
        {
            const auto* in1 = buffer.getReadPointer (1);
            for (int n = 0; n < numSamples; ++n)
                out0[n] = 0.5f * (out0[n] + in1[n]);
        }

        resampler[0].process (out0, out0, numSamples, [stepChannelSample] (float s) {
            return stepChannelSample (0, s);
        });

        for (int ch = 1; ch < numChannels; ++ch)
            buffer.copyFrom (ch, 0, out0, numSamples);
    }
    else
    {
        auto* ch0 = buffer.getWritePointer (0);
        auto* ch1 = buffer.getWritePointer (1);

        resampler[0].process (ch0, ch0, numSamples, [stepChannelSample] (float s) {
            return stepChannelSample (0, s);
        });
        resampler[1].process (ch1, ch1, numSamples, [stepChannelSample] (float s) {
            return stepChannelSample (1, s);
        });
    }
}

// ===========================================================================
// State
// ===========================================================================

float PluginProcessor::getInputDb() const noexcept
{
    return smoothedParams[(size_t) kInput];
}

float PluginProcessor::getOutputDb() const noexcept
{
    return smoothedParams[(size_t) kOutput];
}

float PluginProcessor::getMaster1() const noexcept
{
    return smoothedParams[(size_t) kMaster1];
}

bool PluginProcessor::isStereo() const noexcept
{
    return targets[(size_t) kStereoSwitch].load (std::memory_order_relaxed) > 0.5f;
}

void PluginProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    if (auto xml = parameters.copyState().createXml())
        copyXmlToBinary (*xml, destData);
}

void PluginProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary (data, sizeInBytes))
    {
        const auto tree = juce::ValueTree::fromXml (*xml);

        if (tree.hasType (parameters.state.getType()))
        {
            parameters.replaceState (tree);

            // Preset loads always start from a zero recurrent state, so the same
            // preset always produces the same audio. Without this the hidden state
            // carries over from whatever was playing, and a preset change is not
            // reproducible.
            resetModelState();
        }
    }
}

// ===========================================================================
// Editor
// ===========================================================================

juce::AudioProcessorEditor* PluginProcessor::createEditor()
{
    return new PluginEditor (*this);
}
} // namespace Crunchypy

// The JUCE plugin client declares this with C++ linkage at global scope and looks
// it up by that exact symbol, so it must not sit inside the Crunchypy namespace:
// a namespaced definition mangles to a different name and the VST3 link fails with
// an unresolved external.
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new Crunchypy::PluginProcessor();
}

// Add model.json to BINARYDATA SOURCES in CMakeLists.txt.
// The embedded model must match the 13-feature, 96-hidden geometry in Plugin.h.
