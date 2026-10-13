#pragma once

#include "Plugin.h"

namespace Crunchypy
{

// ===========================================================================
// Editor
//
// Front panel layout matching the Mesa Boogie Mark IIC+:
//   1. Seven Preamp knobs: Volume 1, Treble, Bass, Middle, Master 1, Lead Drive, Lead Master
//   2. Graphic EQ: Five vertical sliders (80, 240, 750, 2200, 6600)
//   3. Graphic EQ Switch: Toggle switch (EQ In / Out)
//   4. Output level knob
//
// The six Mark IIC+ Pull switches live on their respective preamp knobs:
// clicking a knob toggles its pull (an amber ring lights up while pulled).
// ===========================================================================
class PluginEditor : public juce::AudioProcessorEditor,
                     private juce::Timer
{
public:
    explicit PluginEditor (PluginProcessor&);
    ~PluginEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

    static constexpr int kNumPreampKnobs = 7;
    static constexpr int kNumEqBands = 5;

private:
    static constexpr int kMargin = 14;
    static constexpr int kHeaderHeight = 34;
    static constexpr int kStatusHeight = 44;
    static constexpr int kPreferredHeight = 360;
    static constexpr int kPreferredWidth = 1180;

    PluginProcessor& processor;

    // -- Look and Feel ------------------------------------------------------
    class LookAndFeelCustom : public juce::LookAndFeel_V4
    {
    public:
        LookAndFeelCustom();
        void drawRotarySlider (juce::Graphics&, int x, int y, int width, int height,
                               float sliderPosProportional, float rotaryStartAngle,
                               float rotaryEndAngle, juce::Slider&) override;
        void drawLinearSlider (juce::Graphics&, int x, int y, int width, int height,
                               float sliderPos, float minSliderPos, float maxSliderPos,
                               const juce::Slider::SliderStyle, juce::Slider&) override;
        void drawToggleButton (juce::Graphics&, juce::ToggleButton&,
                               bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) override;
    };

    LookAndFeelCustom customLook;

    // -- Input Knob (Utility Trim) ------------------------------------------
    juce::Slider inputKnob;
    juce::Label inputLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> inputAttachment;

    // -- Preamp Stage Header & Knobs ----------------------------------------
    juce::Label preampTitle;
    juce::ToggleButton preampSwitchButton;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> preampSwitchAttachment;

    std::array<juce::Slider, kNumPreampKnobs> preampKnobs;
    std::array<juce::Label, kNumPreampKnobs> preampLabels;
    std::array<juce::Label, kNumPreampKnobs> pullLabels;
    std::array<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>, kNumPreampKnobs> preampAttachments;

    static constexpr int kPullForPreamp[kNumPreampKnobs]
    {
        kPullBrightVol1,       // VOLUME 1
        kPullShiftTreble,      // TREBLE
        kPullShiftBass,        // BASS
        kPullShiftMod,         // MIDDLE
        kPullDeepMaster1,      // MASTER 1
        -1,                    // LEAD DRIVE (no pull)
        kPullBrightLeadMaster  // LEAD MASTER
    };

    std::array<bool, kNumPreampKnobs> pullOn {};
    int pressKnob = -1;

    void mouseDown (const juce::MouseEvent&) override;
    void mouseUp (const juce::MouseEvent&) override;
    int preampIndexFor (const juce::Component*) const noexcept;
    void togglePull (int preampIndex);
    void refreshPullIndicators();

    // -- Graphic EQ Sliders & Header ----------------------------------------
    std::array<juce::Slider, kNumEqBands> eqSliders;
    std::array<juce::Label, kNumEqBands> eqLabels;
    std::array<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>, kNumEqBands> eqAttachments;

    juce::Label eqTitle;
    juce::ToggleButton eqSwitchButton;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> eqSwitchAttachment;

    // -- Poweramp Stage Header & Presence Knob ------------------------------
    juce::Label powerampTitle;
    juce::ToggleButton powerampSwitchButton;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> powerampSwitchAttachment;

    juce::Slider presenceKnob;
    juce::Label presenceLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> presenceAttachment;

    // -- Output Knob (Master Output Trim) & Safety Limiter -----------------
    juce::ToggleButton limiterButton;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> limiterAttachment;

    juce::Slider outputKnob;
    juce::Label outputLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> outputAttachment;

    // -- Header & Bottom Bar ------------------------------------------------
    juce::Label title;
    juce::ToggleButton stereoButton;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> stereoAttachment;
    juce::ToggleButton bypassButton;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> bypassAttachment;

    juce::Label creditLabel;
    juce::Label sampleRateLabel;

    void refreshStatus();
    void timerCallback() override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginEditor)
};

} // namespace Crunchypy