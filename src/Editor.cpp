#include "Editor.h"

namespace Crunchypy
{

namespace
{
inline const juce::Colour kPanel { 0xff1c1e20 };
inline const juce::Colour kPanelEdge { 0xff000000 };
inline const juce::Colour kEqBox { 0xff141517 };
inline const juce::Colour kText { 0xffd8d4c8 };
inline const juce::Colour kTextDim { 0xff8a8578 };
inline const juce::Colour kKnobBody { 0xff2b2e31 };
inline const juce::Colour kKnobPointer { 0xffc8a24a };
inline const juce::Colour kArc { 0xff7d8f6a };
inline const juce::Colour kArcOff { 0xff33363a };
inline const juce::Colour kBad { 0xffd4705a };

// Mesa Mark IIC+ faceplate labels for the 7 preamp knobs
static const char* kPreampNames[PluginEditor::kNumPreampKnobs]
{
    "VOLUME 1", "TREBLE", "BASS", "MIDDLE",
    "MASTER 1", "LEAD DRIVE", "LEAD MASTER"
};

// Engraved pull switch captions above the preamp knobs
static const char* kPullCaptions[PluginEditor::kNumPreampKnobs]
{
    "PULL BRIGHT", "PULL SHIFT", "PULL SHIFT", "PULL ++",
    "PULL DEEP",   "",           "PULL BRIGHT"
};

static const KnobParam kPreampParamEnums[PluginEditor::kNumPreampKnobs]
{
    kVol1, kTreble, kBass, kMiddle,
    kMaster1, kLeadDrive, kLeadMaster
};

// Mesa Mark IIC+ faceplate labels for the 5 Graphic EQ sliders
static const char* kEqFaceplateLabels[PluginEditor::kNumEqBands]
{
    "80", "240", "750", "2200", "6600"
};

static const KnobParam kEqParamEnums[PluginEditor::kNumEqBands]
{
    kEq80, kEq240, kEq750, kEq2200, kEq6600
};
} // namespace

// ===========================================================================
// LookAndFeelCustom
// ===========================================================================

PluginEditor::LookAndFeelCustom::LookAndFeelCustom()
{
    setColour (juce::ResizableWindow::backgroundColourId, juce::Colours::transparentBlack);
}

void PluginEditor::LookAndFeelCustom::drawRotarySlider (juce::Graphics& g,
                                                        int x, int y, int width, int height,
                                                        float sliderPosProportional,
                                                        float rotaryStartAngle,
                                                        float rotaryEndAngle,
                                                        juce::Slider& slider)
{
    const auto bounds = juce::Rectangle<int> (x, y, width, height).toFloat().reduced (4.0f);
    const auto centre = bounds.getCentre();
    const auto radius = juce::jmin (bounds.getWidth(), bounds.getHeight()) * 0.5f - 6.0f;
    if (radius <= 2.0f)
        return;

    const auto thickness = juce::jmax (2.5f, radius * 0.16f);
    const auto angle = rotaryStartAngle
                     + sliderPosProportional * (rotaryEndAngle - rotaryStartAngle);

    // Track arc
    juce::Path track;
    track.addCentredArc (centre.x, centre.y, radius - thickness * 0.5f,
                         radius - thickness * 0.5f, 0.0f,
                         rotaryStartAngle, rotaryEndAngle, true);
    g.setColour (kArcOff);
    g.strokePath (track, juce::PathStrokeType (thickness, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    if (sliderPosProportional > 0.0f)
    {
        juce::Path fill;
        fill.addCentredArc (centre.x, centre.y, radius - thickness * 0.5f,
                            radius - thickness * 0.5f, 0.0f,
                            rotaryStartAngle, angle, true);
        g.setColour (kArc);
        g.strokePath (fill, juce::PathStrokeType (thickness, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    }

    // Body
    g.setColour (kKnobBody);
    g.fillEllipse (centre.x - radius * 0.78f, centre.y - radius * 0.78f,
                   radius * 1.56f, radius * 1.56f);
    g.setColour (kPanelEdge);
    g.drawEllipse (centre.x - radius * 0.78f, centre.y - radius * 0.78f,
                   radius * 1.56f, radius * 1.56f, 1.0f);

    // Pointer
    juce::Path pointer;
    const auto pointerLength = radius * 0.62f;
    const auto pointerWidth = juce::jmax (1.5f, radius * 0.10f);
    pointer.addRoundedRectangle (-pointerWidth * 0.5f, -pointerLength,
                                 pointerWidth, pointerLength * 0.8f,
                                 pointerWidth * 0.5f);
    pointer.applyTransform (juce::AffineTransform::rotation (angle).translated (centre));
    g.setColour (kKnobPointer);
    g.fillPath (pointer);

    // Pulled indicator: amber lit ring around knob body while pulled
    if (slider.getProperties()["pulled"] == juce::var (true))
    {
        g.setColour (kKnobPointer);
        g.drawEllipse (centre.x - radius * 0.78f - 3.0f, centre.y - radius * 0.78f - 3.0f,
                       radius * 1.56f + 6.0f, radius * 1.56f + 6.0f, 2.0f);
    }
}

void PluginEditor::LookAndFeelCustom::drawLinearSlider (juce::Graphics& g,
                                                        int x, int y, int width, int height,
                                                        float sliderPos,
                                                        float minSliderPos,
                                                        float maxSliderPos,
                                                        const juce::Slider::SliderStyle,
                                                        juce::Slider&)
{
    const auto bounds = juce::Rectangle<int> (x, y, width, height).toFloat();
    const float cx = bounds.getCentreX();

    // Vertical slot / groove
    const float trackWidth = 5.0f;
    const float trackTop = bounds.getY() + 10.0f;
    const float trackBottom = bounds.getBottom() - 10.0f;
    const float trackHeight = trackBottom - trackTop;
    const auto trackRect = juce::Rectangle<float> (cx - trackWidth * 0.5f, trackTop, trackWidth, trackHeight);

    g.setColour (juce::Colour (0xff101113));
    g.fillRoundedRectangle (trackRect, 2.5f);
    g.setColour (juce::Colour (0xff050506));
    g.drawRoundedRectangle (trackRect, 2.5f, 1.0f);

    // Center 0 dB reference line
    const float midY = trackTop + trackHeight * 0.5f;
    g.setColour (kTextDim.withAlpha (0.45f));
    g.drawLine (cx - 13.0f, midY, cx + 13.0f, midY, 1.0f);

    // Tick marks at +12 dB and -12 dB
    g.setColour (kTextDim.withAlpha (0.22f));
    g.drawLine (cx - 8.0f, trackTop + trackHeight * 0.25f, cx + 8.0f, trackTop + trackHeight * 0.25f, 1.0f);
    g.drawLine (cx - 8.0f, trackTop + trackHeight * 0.75f, cx + 8.0f, trackTop + trackHeight * 0.75f, 1.0f);

    // Rectangular slider cap
    const float thumbWidth = 24.0f;
    const float thumbHeight = 12.0f;
    const float thumbY = sliderPos - thumbHeight * 0.5f;
    const auto thumbRect = juce::Rectangle<float> (cx - thumbWidth * 0.5f, thumbY, thumbWidth, thumbHeight);

    // Metallic slider cap body
    g.setColour (kKnobBody);
    g.fillRoundedRectangle (thumbRect, 2.0f);
    g.setColour (juce::Colour (0xff484b50));
    g.drawRoundedRectangle (thumbRect, 2.0f, 1.0f);

    // Bright gold center indicator line on cap
    g.setColour (kKnobPointer);
    g.drawLine (cx - thumbWidth * 0.5f + 3.0f, sliderPos, cx + thumbWidth * 0.5f - 3.0f, sliderPos, 2.0f);
}

void PluginEditor::LookAndFeelCustom::drawToggleButton (juce::Graphics& g,
                                                        juce::ToggleButton& button,
                                                        bool, bool)
{
    const auto bounds = button.getLocalBounds().toFloat();
    const bool on = button.getToggleState();

    const auto switchArea = bounds.reduced (2.0f);
    g.setColour (on ? juce::Colour (0xff25282a) : juce::Colour (0xff1b1d1f));
    g.fillRoundedRectangle (switchArea, 4.0f);

    g.setColour (on ? kKnobPointer : juce::Colour (0xff383b3e));
    g.drawRoundedRectangle (switchArea, 4.0f, on ? 1.5f : 1.0f);

    g.setColour (on ? kKnobPointer : kTextDim);
    g.setFont (juce::Font (11.0f, juce::Font::bold));
    g.drawFittedText (button.getButtonText(), switchArea.toNearestInt(), juce::Justification::centred, 1);
}

// ===========================================================================
// PluginEditor
// ===========================================================================

PluginEditor::PluginEditor (PluginProcessor& p)
    : AudioProcessorEditor (&p), processor (p)
{
    // 0. Input Utility Knob (Far Left)
    inputKnob.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    inputKnob.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 65, 18);
    inputKnob.setTextValueSuffix (" dB");
    inputKnob.setDoubleClickReturnValue (true, 0.0);
    inputKnob.setLookAndFeel (&customLook);

    inputAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (
        processor.getState(), PluginProcessor::kParamId[kInput], inputKnob);

    inputKnob.setColour (juce::Slider::textBoxTextColourId, kText);
    inputKnob.setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    inputKnob.setColour (juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);

    inputLabel.setText ("INPUT", juce::dontSendNotification);
    inputLabel.setJustificationType (juce::Justification::centred);
    inputLabel.setColour (juce::Label::textColourId, kText);
    inputLabel.setFont (juce::Font (12.0f, juce::Font::bold));
    addAndMakeVisible (inputLabel);
    addAndMakeVisible (inputKnob);

    // 1. Preamp Stage Header & Knobs
    preampTitle.setText ("PREAMP", juce::dontSendNotification);
    preampTitle.setJustificationType (juce::Justification::centredLeft);
    preampTitle.setColour (juce::Label::textColourId, kText);
    preampTitle.setFont (juce::Font (12.0f, juce::Font::bold));
    addAndMakeVisible (preampTitle);

    preampSwitchButton.setButtonText ("PREAMP IN");
    preampSwitchButton.setLookAndFeel (&customLook);
    preampSwitchAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (
        processor.getState(), PluginProcessor::kParamId[kPreampSwitch], preampSwitchButton);
    addAndMakeVisible (preampSwitchButton);

    for (int i = 0; i < kNumPreampKnobs; ++i)
    {
        auto& knob = preampKnobs[(size_t) i];
        knob.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
        knob.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 65, 18);
        knob.setDoubleClickReturnValue (true, 0.5);
        knob.setLookAndFeel (&customLook);

        const auto paramId = PluginProcessor::kParamId[kPreampParamEnums[i]];
        preampAttachments[(size_t) i] = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (
            processor.getState(), juce::String (paramId), knob);

        knob.setColour (juce::Slider::textBoxTextColourId, kText);
        knob.setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        knob.setColour (juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);

        preampLabels[(size_t) i].setText (kPreampNames[i], juce::dontSendNotification);
        preampLabels[(size_t) i].setJustificationType (juce::Justification::centred);
        preampLabels[(size_t) i].setColour (juce::Label::textColourId, kText);
        preampLabels[(size_t) i].setFont (juce::Font (12.0f, juce::Font::bold));
        addAndMakeVisible (preampLabels[(size_t) i]);

        pullLabels[(size_t) i].setText (kPullCaptions[i], juce::dontSendNotification);
        pullLabels[(size_t) i].setJustificationType (juce::Justification::centred);
        pullLabels[(size_t) i].setColour (juce::Label::textColourId, kTextDim);
        pullLabels[(size_t) i].setFont (juce::Font (11.0f, juce::Font::bold));
        addAndMakeVisible (pullLabels[(size_t) i]);

        addAndMakeVisible (knob);
        knob.addMouseListener (this, false);
    }

    refreshPullIndicators();

    // 2. Five Graphic EQ Vertical Sliders (Faceplate labels: 80, 240, 750, 2200, 6600)
    for (int i = 0; i < kNumEqBands; ++i)
    {
        auto& slider = eqSliders[(size_t) i];
        slider.setSliderStyle (juce::Slider::LinearVertical);
        slider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 42, 16);
        slider.setDoubleClickReturnValue (true, 0.5);
        slider.setLookAndFeel (&customLook);

        const auto paramId = PluginProcessor::kParamId[kEqParamEnums[i]];
        eqAttachments[(size_t) i] = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (
            processor.getState(), juce::String (paramId), slider);

        slider.setColour (juce::Slider::textBoxTextColourId, kTextDim);
        slider.setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        slider.setColour (juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);

        eqLabels[(size_t) i].setText (kEqFaceplateLabels[i], juce::dontSendNotification);
        eqLabels[(size_t) i].setJustificationType (juce::Justification::centred);
        eqLabels[(size_t) i].setColour (juce::Label::textColourId, kText);
        eqLabels[(size_t) i].setFont (juce::Font (12.0f, juce::Font::bold));
        addAndMakeVisible (eqLabels[(size_t) i]);

        addAndMakeVisible (slider);
    }

    eqTitle.setText ("GRAPHIC EQ", juce::dontSendNotification);
    eqTitle.setJustificationType (juce::Justification::centredLeft);
    eqTitle.setColour (juce::Label::textColourId, kText);
    eqTitle.setFont (juce::Font (12.0f, juce::Font::bold));
    addAndMakeVisible (eqTitle);

    // Graphic EQ Switch (EQ IN / OUT)
    eqSwitchButton.setButtonText ("EQ IN");
    eqSwitchButton.setLookAndFeel (&customLook);
    eqSwitchAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (
        processor.getState(), PluginProcessor::kParamId[kEqSwitch], eqSwitchButton);
    addAndMakeVisible (eqSwitchButton);

    // 3. Poweramp Stage Header & Presence Knob
    powerampTitle.setText ("POWERAMP", juce::dontSendNotification);
    powerampTitle.setJustificationType (juce::Justification::centredLeft);
    powerampTitle.setColour (juce::Label::textColourId, kText);
    powerampTitle.setFont (juce::Font (12.0f, juce::Font::bold));
    addAndMakeVisible (powerampTitle);

    powerampSwitchButton.setButtonText ("PWR IN");
    powerampSwitchButton.setLookAndFeel (&customLook);
    powerampSwitchAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (
        processor.getState(), PluginProcessor::kParamId[kPowerampSwitch], powerampSwitchButton);
    addAndMakeVisible (powerampSwitchButton);

    presenceKnob.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    presenceKnob.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 65, 18);
    presenceKnob.setDoubleClickReturnValue (true, 0.5);
    presenceKnob.setLookAndFeel (&customLook);

    presenceAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (
        processor.getState(), PluginProcessor::kParamId[kPresence], presenceKnob);

    presenceKnob.setColour (juce::Slider::textBoxTextColourId, kText);
    presenceKnob.setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    presenceKnob.setColour (juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);

    presenceLabel.setText ("PRESENCE", juce::dontSendNotification);
    presenceLabel.setJustificationType (juce::Justification::centred);
    presenceLabel.setColour (juce::Label::textColourId, kText);
    presenceLabel.setFont (juce::Font (12.0f, juce::Font::bold));
    addAndMakeVisible (presenceLabel);
    addAndMakeVisible (presenceKnob);

    // 4. Output Rotary Knob (Far Right) & Safety Limiter
    limiterButton.setButtonText ("LIMITER");
    limiterButton.setLookAndFeel (&customLook);
    limiterAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (
        processor.getState(), PluginProcessor::kParamId[kLimiterSwitch], limiterButton);
    addAndMakeVisible (limiterButton);

    outputKnob.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    outputKnob.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 65, 18);
    outputKnob.setTextValueSuffix (" dB");
    outputKnob.setDoubleClickReturnValue (true, 0.0);
    outputKnob.setLookAndFeel (&customLook);

    outputAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (
        processor.getState(), PluginProcessor::kParamId[kOutput], outputKnob);

    outputKnob.setColour (juce::Slider::textBoxTextColourId, kText);
    outputKnob.setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    outputKnob.setColour (juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);

    outputLabel.setText ("OUTPUT", juce::dontSendNotification);
    outputLabel.setJustificationType (juce::Justification::centred);
    outputLabel.setColour (juce::Label::textColourId, kText);
    outputLabel.setFont (juce::Font (12.0f, juce::Font::bold));
    addAndMakeVisible (outputLabel);
    addAndMakeVisible (outputKnob);

    // 5. Header Controls (CRUNCHYPY, STEREO, BYPASS)
    title.setText ("CRUNCHYPY", juce::dontSendNotification);
    title.setJustificationType (juce::Justification::centredLeft);
    title.setColour (juce::Label::textColourId, kText);
    title.setFont (juce::Font (18.0f, juce::Font::bold));
    addAndMakeVisible (title);

    stereoButton.setButtonText (processor.isStereo() ? "STEREO" : "MONO");
    stereoButton.setLookAndFeel (&customLook);
    stereoButton.onClick = [this]
    {
        stereoButton.setButtonText (stereoButton.getToggleState() ? "STEREO" : "MONO");
    };
    stereoAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (
        processor.getState(), PluginProcessor::kParamId[kStereoSwitch], stereoButton);
    addAndMakeVisible (stereoButton);

    bypassButton.setButtonText ("BYPASS");
    bypassButton.setLookAndFeel (&customLook);
    bypassAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (
        processor.getState(), kBypassParamId, bypassButton);
    addAndMakeVisible (bypassButton);

    // 6. Bottom Status Bar (Credit on left, Sample rate / Engine on right)
    creditLabel.setText (juce::String ("CRUNCHYPY ") + juce::String::charToString (0x00B7) + " Designed by Yahia Kemari",
                         juce::dontSendNotification);
    creditLabel.setJustificationType (juce::Justification::centredLeft);
    creditLabel.setColour (juce::Label::textColourId, kTextDim);
    creditLabel.setFont (juce::Font (11.0f, juce::Font::bold));
    addAndMakeVisible (creditLabel);

    sampleRateLabel.setJustificationType (juce::Justification::centredRight);
    sampleRateLabel.setColour (juce::Label::textColourId, kTextDim);
    sampleRateLabel.setFont (juce::Font (11.0f));
    addAndMakeVisible (sampleRateLabel);

    refreshStatus();
    startTimer (500);

    setResizable (true, true);
    setResizeLimits (900, 320, 1500, 600);
    setSize (kPreferredWidth, kPreferredHeight);
}

PluginEditor::~PluginEditor()
{
    stopTimer();

    for (auto& knob : preampKnobs)
    {
        knob.removeMouseListener (this);
        knob.setLookAndFeel (nullptr);
    }
    for (auto& slider : eqSliders)
        slider.setLookAndFeel (nullptr);

    inputKnob.setLookAndFeel (nullptr);
    presenceKnob.setLookAndFeel (nullptr);
    outputKnob.setLookAndFeel (nullptr);

    preampSwitchButton.setLookAndFeel (nullptr);
    eqSwitchButton.setLookAndFeel (nullptr);
    powerampSwitchButton.setLookAndFeel (nullptr);
    limiterButton.setLookAndFeel (nullptr);
    stereoButton.setLookAndFeel (nullptr);
    bypassButton.setLookAndFeel (nullptr);
}

void PluginEditor::paint (juce::Graphics& g)
{
    g.fillAll (kPanel);

    // Inner bevel
    g.setColour (kPanelEdge.withAlpha (0.6f));
    g.drawRect (getLocalBounds(), 1);

    // Section dividers
    g.setColour (juce::Colour (0xff121315));
    const auto contentTop = kMargin + kHeaderHeight + 4;
    const auto contentBottom = getHeight() - kMargin - 24;

    if (preampTitle.getX() > 0)
    {
        const int div1 = preampTitle.getX() - 5;
        g.drawLine ((float) div1, (float) contentTop, (float) div1, (float) contentBottom, 1.0f);
    }
    if (eqTitle.getX() > 0)
    {
        const int div2 = eqTitle.getX() - 5;
        g.drawLine ((float) div2, (float) contentTop, (float) div2, (float) contentBottom, 1.0f);
    }
    if (powerampTitle.getX() > 0)
    {
        const int div3 = powerampTitle.getX() - 5;
        g.drawLine ((float) div3, (float) contentTop, (float) div3, (float) contentBottom, 1.0f);
    }
    if (outputLabel.getX() > 0)
    {
        const int div4 = outputLabel.getX() - 5;
        g.drawLine ((float) div4, (float) contentTop, (float) div4, (float) contentBottom, 1.0f);
    }

    // Graphic EQ Bezel box (draw a darker recessed area behind the EQ sliders)
    if (eqSliders[0].getWidth() > 0)
    {
        const int left = eqSliders[0].getX() - 6;
        const int right = eqSliders[kNumEqBands - 1].getRight() + 6;
        const int top = eqSliders[0].getY() - 4;
        const int bottom = eqLabels[0].getBottom() + 4;
        const auto eqBounds = juce::Rectangle<int> (left, top, right - left, bottom - top);

        g.setColour (kEqBox);
        g.fillRoundedRectangle (eqBounds.toFloat(), 4.0f);
        g.setColour (juce::Colour (0xff0a0b0d));
        g.drawRoundedRectangle (eqBounds.toFloat(), 4.0f, 1.2f);
    }
}

void PluginEditor::resized()
{
    auto area = getLocalBounds().reduced (kMargin);

    // 1. Top Header
    auto header = area.removeFromTop (kHeaderHeight);
    bypassButton.setBounds (header.removeFromRight (76).withSizeKeepingCentre (72, 22));
    header.removeFromRight (6);
    stereoButton.setBounds (header.removeFromRight (76).withSizeKeepingCentre (72, 22));
    title.setBounds (header);

    // 2. Bottom Status Bar
    auto footer = area.removeFromBottom (20);
    creditLabel.setBounds (footer.removeFromLeft (footer.getWidth() / 2));
    sampleRateLabel.setBounds (footer);

    area.removeFromTop (6);
    area.removeFromBottom (6);

    // 3. Stage Columns Layout:
    // [ INPUT ] | [ PREAMP (7 Knobs) ] | [ GRAPHIC EQ (5 Sliders) ] | [ POWERAMP (Presence) ] | [ OUTPUT ]
    const int totalW = area.getWidth();
    const int inputW    = juce::jlimit (60, 90, static_cast<int> (totalW * 0.075f));
    const int outputW   = juce::jlimit (60, 90, static_cast<int> (totalW * 0.075f));
    const int eqW       = juce::jlimit (180, 280, static_cast<int> (totalW * 0.235f));
    const int powerampW = juce::jlimit (120, 160, static_cast<int> (totalW * 0.125f));
    const int gapW = 8;
    const int preampW   = totalW - (inputW + outputW + eqW + powerampW + 4 * gapW);

    auto inputArea = area.removeFromLeft (inputW);
    area.removeFromLeft (gapW);

    auto preampArea = area.removeFromLeft (preampW);
    area.removeFromLeft (gapW);

    auto eqArea = area.removeFromLeft (eqW);
    area.removeFromLeft (gapW);

    auto powerampArea = area.removeFromLeft (powerampW);
    area.removeFromLeft (gapW);

    auto outputArea = area;

    // Section Header row height
    constexpr int kSectionHeaderH = 24;

    // Preamp Stage Header
    auto preampTop = preampArea.removeFromTop (kSectionHeaderH);
    preampSwitchButton.setBounds (preampTop.removeFromRight (86).withSizeKeepingCentre (82, 22));
    preampTitle.setBounds (preampTop);
    preampArea.removeFromTop (4);

    // Graphic EQ Stage Header
    auto eqTop = eqArea.removeFromTop (kSectionHeaderH);
    eqSwitchButton.setBounds (eqTop.removeFromRight (64).withSizeKeepingCentre (60, 22));
    eqTitle.setBounds (eqTop);
    eqArea.removeFromTop (4);

    // Poweramp Stage Header
    auto powerampTop = powerampArea.removeFromTop (kSectionHeaderH);
    powerampSwitchButton.setBounds (powerampTop.removeFromRight (80).withSizeKeepingCentre (76, 22));
    powerampTitle.setBounds (powerampTop);
    powerampArea.removeFromTop (4);

    // Match top margins of input and output to the stage headers
    inputArea.removeFromTop (kSectionHeaderH + 4);

    auto outputTop = outputArea.removeFromTop (kSectionHeaderH);
    limiterButton.setBounds (outputTop.withSizeKeepingCentre (74, 22));
    outputArea.removeFromTop (4);

    // Unified knob dimensions across all sections
    const float preampSlotW = preampArea.getWidth() / static_cast<float> (kNumPreampKnobs);
    constexpr int pullH = 16;
    constexpr int nameH = 18;
    constexpr int textBoxH = 18;

    const int availableH = preampArea.getHeight();
    const int knobDiam = juce::jlimit (36, 74, juce::jmin (static_cast<int> (preampSlotW) - 8,
                                                           availableH - pullH - nameH - textBoxH));
    const int knobCompH = knobDiam + textBoxH;
    const int blockH = pullH + knobCompH + nameH;
    const int blockY = preampArea.getY() + juce::jmax (0, (availableH - blockH) / 2);

    // 1. Layout Preamp Knobs (7)
    for (int i = 0; i < kNumPreampKnobs; ++i)
    {
        const int slotX = preampArea.getX() + static_cast<int> (i * preampSlotW);
        const int slotW = static_cast<int> (preampSlotW);

        pullLabels[(size_t) i].setBounds (slotX, blockY, slotW, pullH);
        preampKnobs[(size_t) i].setBounds (slotX + (slotW - knobDiam) / 2, blockY + pullH, knobDiam, knobCompH);
        preampLabels[(size_t) i].setBounds (slotX, blockY + pullH + knobCompH, slotW, nameH);
    }

    // 2. Layout Input Knob
    inputKnob.setBounds (inputArea.getX() + (inputArea.getWidth() - knobDiam) / 2, blockY + pullH, knobDiam, knobCompH);
    inputLabel.setBounds (inputArea.getX(), blockY + pullH + knobCompH, inputArea.getWidth(), nameH);

    // 3. Layout Presence Knob (Poweramp)
    presenceKnob.setBounds (powerampArea.getX() + (powerampArea.getWidth() - knobDiam) / 2, blockY + pullH, knobDiam, knobCompH);
    presenceLabel.setBounds (powerampArea.getX(), blockY + pullH + knobCompH, powerampArea.getWidth(), nameH);

    // 4. Layout Output Knob
    outputKnob.setBounds (outputArea.getX() + (outputArea.getWidth() - knobDiam) / 2, blockY + pullH, knobDiam, knobCompH);
    outputLabel.setBounds (outputArea.getX(), blockY + pullH + knobCompH, outputArea.getWidth(), nameH);

    // 5. Layout Graphic EQ Sliders (5) - Tall Mesa Boogie style
    const float eqSlotW = eqArea.getWidth() / static_cast<float> (kNumEqBands);
    const int eqLabelY = blockY + pullH + knobCompH;
    const int eqSliderTopY = eqArea.getY() + 4;
    const int eqSliderH = eqLabelY - eqSliderTopY - 2;

    for (int i = 0; i < kNumEqBands; ++i)
    {
        const int slotX = eqArea.getX() + static_cast<int> (i * eqSlotW);
        const int slotW = static_cast<int> (eqSlotW);

        eqSliders[(size_t) i].setBounds (slotX, eqSliderTopY, slotW, eqSliderH);
        eqLabels[(size_t) i].setBounds (slotX, eqLabelY, slotW, nameH);
    }
}

// ===========================================================================
// Click-to-pull
// ===========================================================================

int PluginEditor::preampIndexFor (const juce::Component* c) const noexcept
{
    if (c == nullptr)
        return -1;

    for (int i = 0; i < kNumPreampKnobs; ++i)
        if (c == &preampKnobs[(size_t) i])
            return i;

    return -1;
}

void PluginEditor::mouseDown (const juce::MouseEvent& e)
{
    pressKnob = preampIndexFor (e.eventComponent);
}

void PluginEditor::mouseUp (const juce::MouseEvent& e)
{
    const int idx = preampIndexFor (e.eventComponent);
    if (idx < 0 || idx != pressKnob)
    {
        pressKnob = -1;
        return;
    }
    pressKnob = -1;

    if (e.mouseWasDraggedSinceMouseDown())
        return;

    togglePull (idx);
}

void PluginEditor::togglePull (int preampIndex)
{
    const auto pull = kPullForPreamp[(size_t) preampIndex];
    if (pull < 0)
        return;

    auto* param = processor.getState().getParameter (juce::String (PluginProcessor::kParamId[pull]));
    if (param == nullptr)
        return;

    param->setValueNotifyingHost (param->getValue() < 0.5f ? 1.0f : 0.0f);
    refreshPullIndicators();
}

void PluginEditor::refreshPullIndicators()
{
    for (int i = 0; i < kNumPreampKnobs; ++i)
    {
        const auto pull = kPullForPreamp[(size_t) i];
        bool on = false;
        if (pull >= 0)
            if (auto* param = processor.getState().getParameter (juce::String (PluginProcessor::kParamId[pull])))
                on = param->getValue() >= 0.5f;

        if (pullOn[(size_t) i] != on)
        {
            pullOn[(size_t) i] = on;
            preampKnobs[(size_t) i].getProperties().set ("pulled", juce::var (on));
            preampKnobs[(size_t) i].repaint();
        }
    }
}

void PluginEditor::refreshStatus()
{
    stereoButton.setButtonText (stereoButton.getToggleState() ? "STEREO" : "MONO");

    if (! processor.isModelValid())
    {
        sampleRateLabel.setColour (juce::Label::textColourId, kBad);
        sampleRateLabel.setText ("PREAMP MODEL FAILED TO LOAD: " + processor.getModelError(), juce::dontSendNotification);
        return;
    }

    if (! processor.isPowerampValid())
    {
        sampleRateLabel.setColour (juce::Label::textColourId, kBad);
        sampleRateLabel.setText ("POWERAMP MODEL FAILED TO LOAD: " + processor.getPowerampError(), juce::dontSendNotification);
        return;
    }

    juce::String rateText;
    if (processor.isResamplingActive())
        rateText = juce::String (processor.getHostSampleRate(), 0) + " Hz (Resampled to 48 kHz)";
    else
        rateText = "48000 Hz (Native)";

    sampleRateLabel.setColour (juce::Label::textColourId, kTextDim);
    sampleRateLabel.setText (rateText, juce::dontSendNotification);
}

void PluginEditor::timerCallback()
{
    refreshStatus();
    refreshPullIndicators();
}

} // namespace Crunchypy