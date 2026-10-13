#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

#include <RTNeural.h>
#include "EQStage.h"
#include "FixedRateResampler.h"
#include "OutputLimiter.h"

#include <array>
#include <atomic>
#include <vector>

namespace Crunchypy
{

// ===========================================================================
// Network geometry -- Mark IIC+ PREAMP configuration.
//
// These must match the exported model.json metadata exactly. A retrained network
// with a different shape fails loudly at load instead of silently reading past
// the end of a buffer.
//
// MARK IIC+ PREAMP:
//   12 knobs conditioned into the network:
//     Free knobs (continuous 0..1): Vol1, Lead Drive, Treble, Bass, Middle, Lead Master
//     Switches (binary 0/1): Pull Bright (Vol1), Pull Shift (Treble), Pull Shift (Bass),
//                             Pull Shift (++ mod), Pull Deep (Master1), Pull Bright (Lead Master)
//   Pinned (NOT in feature vector):
//     Pull Lead = 1.0 (always ON, lead channel engaged)
//   Master 1 is a UI knob, NOT a conditioning channel: it was pinned at 0.5 during
//   rendering so the network never saw it vary (a constant channel would only waste
//   capacity). The plugin applies it AFTER the network as a voltage divide -- see the
//   Master 1 section below -- and its default position of 0.5 is the render reference.
//
//   Hidden size 96: matches the model.json bundled in this build
//   (13 inputs -> dense 96 -> GRU 96 -> dense 1). If you re-export a wider
//   network, bump kHiddenSize together with the exported weights, or the model
//   fails its geometry validation at load.
// ===========================================================================
inline constexpr int kNumKnobs = 12;
inline constexpr int kHiddenSize = 96;

/** One audio sample plus one value per conditioned knob. */
inline constexpr int kInputSize = 1 + kNumKnobs;
inline constexpr int kOutputSize = 1;

/** Where every knob starts on plugin load / preset. */
inline constexpr float kDefaultKnob = 0.5f;

/** Parameter ID of the bypass toggle, shared by processor and editor. */
inline constexpr const char* kBypassParamId = "bypass";

// ===========================================================================
// Master 1.
//
// Master 1 is the preamp's master volume: a passive output voltage divide, exactly
// like the original Volume knob. It is pinned at 0.5 during rendering, so
// the model's output already contains the Master-1-at-0.5 level and the knob's
// default position of 0.5 is unity. It is a UI knob applied AFTER the network and
// is never conditioned into it (a feature that never varied in training would only
// waste capacity on a constant).
//
//   gain(v) = kMaster1GainAtFull * clamp01(v)  ->  0 at v=0, 1 at v=0.5, 2 at v=1
// ===========================================================================
inline constexpr float kMaster1GainAtFull = 2.0f;

/** Master 1's output ratio for a knob position in [0, 1]. */
inline constexpr float master1GainFor (float v) noexcept
{
    // Clamped by hand rather than with juce::jlimit, which is not constexpr.
    return kMaster1GainAtFull * (v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v));
}

// ===========================================================================
// Output level.
//
// A post-network trim in dB applied on top of the Master 1 divide, so the output
// chain is: model -> Master 1 (voltage divide) -> Output (dB trim).
//
// Range: -24 dB (tame the preamp) to +6 dB (push it). Default 0 dB = unity.
// 0.1 dB steps: finer than anyone can hear but land on round values when typed.
// ===========================================================================
inline constexpr float kOutputMinDb = -24.0f;
inline constexpr float kOutputMaxDb = 6.0f;
inline constexpr float kOutputStepDb = 0.1f;
inline constexpr float kOutputDefaultDb = 0.0f;

/** Convert output knob dB to linear gain. */
inline float outputGainFromDb (float dB) noexcept
{
    return juce::Decibels::decibelsToGain (dB);
}

// ===========================================================================
// Input level.
//
// Pre-network gain trim in dB: -24 dB to +24 dB, default 0 dB (unity).
// ===========================================================================
inline constexpr float kInputMinDb = -24.0f;
inline constexpr float kInputMaxDb = 24.0f;
inline constexpr float kInputStepDb = 0.1f;
inline constexpr float kInputDefaultDb = 0.0f;

/** Convert input knob dB to linear gain. */
inline float inputGainFromDb (float dB) noexcept
{
    return juce::Decibels::decibelsToGain (dB);
}

// ===========================================================================
// Parameters.
//
// The enum values are stable identifiers -- kParamId and saved presets key on
// them -- and deliberately NOT the panel's left-to-right order, which lives in
// the editor's kKnobOrder. The network's conditioning order is KNOB_ORDER below.
// Keeping the three separate means the panel can be rearranged without touching
// the model or invalidating presets.
// ===========================================================================
enum KnobParam
{
    kVol1 = 0,
    kLeadDrive = 1,
    kTreble = 2,
    kBass = 3,
    kMiddle = 4,
    kLeadMaster = 5,
    kPullBrightVol1 = 6,
    kPullShiftTreble = 7,
    kPullShiftBass = 8,
    kPullShiftMod = 9,
    kPullDeepMaster1 = 10,
    kPullBrightLeadMaster = 11,
    kMaster1 = 12,
    kOutput = 13,
    kInput = 14,
    // Graphic EQ knobs (80/240/750/2200/6600 Hz).
    kEq80 = 15,
    kEq240 = 16,
    kEq750 = 17,
    kEq2200 = 18,
    kEq6600 = 19,
    kEqSwitch = 20,
    kPresence = 21,
    kPreampSwitch = 22,
    kPowerampSwitch = 23,
    kStereoSwitch = 24,
    kLimiterSwitch = 25,
    kNumParams = 26
};

/** The Pull switches are contiguous in KnobParam, mirrored by kParamId. The
 *  processor uses the bounds to keep switch channels binary and unsmoothed. */
inline constexpr int kFirstSwitchParam = kPullBrightVol1;
inline constexpr int kLastSwitchParam = kPullBrightLeadMaster;

// ===========================================================================
// Network feature slots.
//
// feature slot 0 = audio sample
// feature slots 1..12 = the 12 knobs in KNOB_ORDER below
//
// KNOB_ORDER is the order the network expects. kFeatureSlot maps each UI
// parameter enum to its slot in the feature vector.
// ===========================================================================
/** Knob order the network is trained on. MUST match model.json metadata["knobs"]. */
inline constexpr const char* KNOB_ORDER[kNumKnobs] {
    "Vol1",                // slot 1
    "Lead Drive",          // slot 2
    "Treble",              // slot 3
    "Bass",                // slot 4
    "Middle",              // slot 5
    "Lead Master",         // slot 6
    "Pull Bright (Vol1)",  // slot 7
    "Pull Shift (Treble)", // slot 8
    "Pull Shift (Bass)",   // slot 9
    "Pull Shift (++ mod)", // slot 10
    "Pull Deep (Master1)", // slot 11
    "Pull Bright (Lead Master)" // slot 12
};

/** Maps each knobs enum to its feature slot index (1-based, slot 0 is audio).
 *  -1 means the parameter is not fed into the network (Master 1, Output). */
inline constexpr std::array<int, kNumParams> kFeatureSlot {
    1,  // kVol1       -> feature slot 1
    2,  // kLeadDrive  -> feature slot 2
    3,  // kTreble     -> feature slot 3
    4,  // kBass       -> feature slot 4
    5,  // kMiddle     -> feature slot 5
    6,  // kLeadMaster -> feature slot 6
    7,  // kPullBrightVol1      -> feature slot 7
    8,  // kPullShiftTreble     -> feature slot 8
    9,  // kPullShiftBass       -> feature slot 9
    10, // kPullShiftMod        -> feature slot 10
    11, // kPullDeepMaster1     -> feature slot 11
    12, // kPullBrightLeadMaster -> feature slot 12
    -1, // kMaster1    -> post-preamp voltage divide / poweramp drive, not in preamp feature vector
    -1, // kOutput     -> post-network gain, not in feature vector
    -1, // kInput      -> pre-network gain, not in feature vector
    -1, -1, -1, -1, -1, // EQ sliders -> independent linear EQ stage, not conditioned
    -1, // kEqSwitch   -> EQ on/off switch, not in feature vector
    -1, // kPresence   -> conditioned into poweramp, not in preamp feature vector
    -1, // kPreampSwitch -> preamp bypass, not in feature vector
    -1, // kPowerampSwitch -> poweramp bypass, not in feature vector
    -1, // kStereoSwitch -> stereo mode, not in feature vector
    -1  // kLimiterSwitch -> safety limiter switch, not in feature vector
};

static_assert (kInputSize == 1 + kNumKnobs,
               "Feature vector must be audio plus one slot per knob.");
static_assert (kFeatureSlot[kOutput] < 0 && kFeatureSlot[kMaster1] < 0 && kFeatureSlot[kInput] < 0,
               "Master 1, Input and Output are post/pre-network gain and must never be conditioned into the network.");

// ===========================================================================
// The networks (Preamp & Poweramp).
//
// model.json declares three layers (dense, gru, dense), but RTNeural needs four:
// layer 0 is a dense with "activation":"tanh", and ModelT::parseJson treats that as
// an assertion that the *next* template layer is the matching activation.
// ===========================================================================
using Network = RTNeural::ModelT<float,
                                 kInputSize,
                                 kOutputSize,
                                 RTNeural::DenseT<float, kInputSize, kHiddenSize>,
                                 RTNeural::TanhActivationT<float, kHiddenSize>,
                                 RTNeural::GRULayerT<float, kHiddenSize, kHiddenSize>,
                                 RTNeural::DenseT<float, kHiddenSize, kOutputSize>>;

inline constexpr int kPowerampInputSize = 2; // [audio, Presence]
inline constexpr int kPowerampHiddenSize = 32;

using PowerampNetwork = RTNeural::ModelT<float,
                                         kPowerampInputSize,
                                         kOutputSize,
                                         RTNeural::DenseT<float, kPowerampInputSize, kPowerampHiddenSize>,
                                         RTNeural::TanhActivationT<float, kPowerampHiddenSize>,
                                         RTNeural::GRULayerT<float, kPowerampHiddenSize, kPowerampHiddenSize>,
                                         RTNeural::DenseT<float, kPowerampHiddenSize, kOutputSize>>;

// ===========================================================================
// AudioProcessor
// ===========================================================================
class PluginProcessor : public juce::AudioProcessor,
                        private juce::AudioProcessorValueTreeState::Listener
{
public:
    PluginProcessor();
    ~PluginProcessor() override;

    // --- lifecycle ---------------------------------------------------------
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

    // --- processing --------------------------------------------------------
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    void processBlockBypassed (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    /** The bypass parameter, so hosts map their own bypass control to it. */
    juce::AudioProcessorParameter* getBypassParameter() const override;

    // --- editor ------------------------------------------------------------
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    // --- identity ----------------------------------------------------------
    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override;

    // --- programs ----------------------------------------------------------
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return "Default"; }
    void changeProgramName (int, const juce::String&) override {}

    // --- state -------------------------------------------------------------
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // --- parameters --------------------------------------------------------
    juce::AudioProcessorValueTreeState& getState() noexcept { return parameters; }

    /** True when the embedded network loaded and validated. */
    bool isModelValid() const noexcept { return modelValid; }

    /** True when the embedded EQ weights loaded and validated. */
    bool isEqValid() const noexcept { return eqStage[0].isLoaded(); }

    /** True when the embedded poweramp network loaded and validated. */
    bool isPowerampValid() const noexcept { return powerampValid; }

    /** Populated only when modelValid is false. */
    const juce::String& getModelError() const noexcept { return modelError; }

    /** Populated only when powerampValid is false. */
    const juce::String& getPowerampError() const noexcept { return powerampError; }

    /** Ground-truth validation ESR carried over from the export, for display. */
    double getExportValEsr() const noexcept { return exportValEsr; }

    /** Ground-truth poweramp validation ESR. */
    double getPowerampValEsr() const noexcept { return powerampValEsr; }

    /** Knob ordering the embedded model expects, for display and for tests. */
    const juce::StringArray& getModelKnobOrder() const noexcept { return modelKnobOrder; }

    /** Returns current host sample rate. */
    double getHostSampleRate() const noexcept { return currentHostRate; }

    /** True when host sample rate differs from the native 48 kHz model training rate. */
    bool isResamplingActive() const noexcept { return needsResampling; }

    /** Returns the input level in dB. */
    float getInputDb() const noexcept;

    /** Returns the output level in dB. */
    float getOutputDb() const noexcept;

    /** Returns the current Master 1 knob position in [0, 1]. */
    float getMaster1() const noexcept;

    /** Returns true when processing in stereo mode. */
    bool isStereo() const noexcept;

    /** Parameter ID strings, indexed by enum KnobParam. Shared with the editor. */
    static constexpr const char* kParamId[kNumParams] {
        "vol1", "leadDrive", "treble", "bass", "middle", "leadMaster",
        "pullBrightVol1", "pullShiftTreble", "pullShiftBass", "pullShiftMod",
        "pullDeepMaster1", "pullBrightLeadMaster", "master1", "output", "input",
        "eq80", "eq240", "eq750", "eq2200", "eq6600", "eqSwitch",
        "presence", "preampSwitch", "powerampSwitch", "stereoSwitch", "limiterSwitch"
    };

private:
    // -- network ------------------------------------------------------------
    /** Declares knobs and DSP parameters. */
    static juce::AudioProcessorValueTreeState::ParameterLayout makeLayout();

    /** Host thread: a parameter moved. Stores to an atomic for the audio thread. */
    void parameterChanged (const juce::String& parameterID, float newValue) override;

    /** Parses and validates the embedded model.json. Off the audio thread. */
    bool loadModel();

    /** Parses and validates the embedded poweramp.json. Off the audio thread. */
    bool loadPowerampModel();

    /** Discards the network's recurrent state. */
    void resetModelState();

    std::array<Network, 2> model;
    bool modelValid = false;
    juce::String modelError;
    juce::StringArray modelKnobOrder;
    double exportValEsr = 0.0;

    std::array<PowerampNetwork, 2> powerampModel;
    bool powerampValid = false;
    juce::String powerampError;
    double powerampValEsr = 0.0;

    // -- parameters ---------------------------------------------------------
    juce::AudioProcessorValueTreeState parameters;

    /** Target values, written by the host thread and read by the audio thread.
     *  All knobs store 0..1 except Output/Input which store dB. */
    std::array<std::atomic<float>, kNumParams> targets;


    friend class PluginEditor;

    // -- Resampling & transparent DC blocking ------------------------------
    std::array<juce::dsp::FirstOrderTPTFilter<float>, 2> dcBlocker;

    /** Output safety limiter (see OutputLimiter.h). Last stage in the chain. */
    std::array<OutputLimiter, 2> limiter;

    /** Graphic EQ stage: per-sample IIR, coefficients re-evaluated on knob change. */
    std::array<EQStage, 2> eqStage;

    /** Per-sample smoothed values for every knob. */
    std::array<float, kNumParams> smoothedParams;

    /** Raw bypass parameter value, host thread writes, audio thread loads. */
    std::atomic<float>* bypassValue = nullptr;

    double currentHostRate = 48000.0;
    bool needsResampling = false;

    std::array<FixedRateResampler, 2> resampler;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginProcessor)
};

} // namespace Crunchypy
